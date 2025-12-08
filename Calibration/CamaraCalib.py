"""
This file will contain camera calibration functions.
This will return a camara matrices for a given set of camaras, using aruco markers.

Need to change following prameters for different setups:
- file_paths: List of paths to calibration images for each camera. Order matters and should correspond to camera numbers.
- size_of_marker: The length of the aruco marker's side in meters.
- output_file: The list names for the files where calibration results will be saved.
    - default: 'calibration_results_camX.npz' with X being the camera number.
- aruco_dict_type: The type of aruco dictionary used for marker detection.
    - default: cv2.aruco.DICT_4X4_50 This is the aruco board available from the makerspace Aruco board.
- max_iterations: The maximum number of iterations for corner refinement.
- termination_eps: The desired accuracy for corner refinement. (this is between consecutive iterations shows convergences)
"""

import os
import sys
import glob
import argparse
from typing import List, Tuple, Optional

import cv2
import numpy as np

# ---------------------------
# User / repository constants
# (kept exactly as provided by you)
# ---------------------------
CALIB_IMG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CalibImg'))
CAMERA_PREFIXES = ['camera_0', 'camera_2']
# Size of the aruco marker (same units used when printing / saving). Current code uses mm.
size_of_marker = 26  # mm
output_file = None  # will be generated per-camera below
aruco_dict_type = cv2.aruco.DICT_4X4_50  # Aruco dictionary type
max_iterations = 1000  # Max iterations for corner refinement
termination_eps = 1e-9  # Desired accuracy for corner refinement
# ---------------------------

# cornerSubPix criteria (uses your max_iterations and termination_eps)
_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, termination_eps)

def detect_markers_wrapper(gray: np.ndarray, aruco_dict, parameters=None):
    """
    Detect markers in an image handling both older and newer OpenCV aruco APIs.

    Returns:
        corners, ids, rejected
    """
    # Try the classic API detectMarkers (present in many OpenCV versions)
    try:
        detect = getattr(cv2.aruco, "detectMarkers", None)
        if detect is not None:
            return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    except Exception:
        # fallback to newer detector below
        pass

    # Try the newer ArucoDetector interface
    try:
        ArucoDetector = getattr(cv2.aruco, "ArucoDetector", None)
        if ArucoDetector is not None:
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, rejected = detector.detectMarkers(gray)
            return corners, ids, rejected
    except Exception:
        pass

    raise RuntimeError("No compatible ArUco detection method available. Ensure opencv-contrib-python is installed.")


def gather_image_paths(root: str, prefix: str) -> List[str]:
    """
    Walks the CALIB_IMG_ROOT searching each subfolder for files that start with prefix.
    Returns a sorted list of image file paths.
    """
    images = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Calib img root not found: {root}")

    # Search subfolders (non-recursive deeper than one level) - adjust if your layout differs
    for sub in sorted(os.listdir(root)):
        subp = os.path.join(root, sub)
        if not os.path.isdir(subp):
            continue
        # look for files starting with prefix in this subfolder
        pattern = os.path.join(subp, f"{prefix}*")
        found = sorted(glob.glob(pattern))
        images.extend(found)
    return images


def camera_calibration_from_image_list(
    image_files: List[str],
    marker_size_mm: float,
    aruco_dict_type=cv2.aruco.DICT_4X4_50
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], float]:
    """
    Calibrate a camera from a list of image file paths that contain ArUco markers.

    Returns:
        camera_matrix, dist_coeffs, image_size (width, height), reprojection_error
    """

    if len(image_files) == 0:
        raise ValueError("No image files supplied for calibration.")

    # Precompute marker 3D corner coordinates in the board's units (mm).
    # We assume each marker is a square of `marker_size_mm` with corners in this order:
    # (0,0), (marker_size, 0), (marker_size, marker_size), (0, marker_size)
    marker_objp = np.array([
        [0.0, 0.0, 0.0],
        [marker_size_mm, 0.0, 0.0],
        [marker_size_mm, marker_size_mm, 0.0],
        [0.0, marker_size_mm, 0.0]
    ], dtype=np.float32)

    # prepare aruco dictionary & parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    # Detector parameters (compatible with both APIs)
    try:
        parameters = cv2.aruco.DetectorParameters_create()
    except Exception:
        parameters = None

    objpoints = []  # list of (N_points, 3) arrays, one per image
    imgpoints = []  # list of (N_points, 2) arrays, one per image
    used_images = 0
    last_img_size = None

    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: failed to read '{fname}', skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        last_img_size = (gray.shape[1], gray.shape[0])  # width, height

        try:
            corners, ids, rejected = detect_markers_wrapper(gray, aruco_dict, parameters)
        except RuntimeError as e:
            print(f"Marker detection failed for OpenCV API: {e}")
            raise

        if ids is None or len(ids) == 0:
            # no markers in this image
            continue

        # For this image: collect all marker corners into one view, and the matching objpoints
        all_img_pts = []
        all_obj_pts = []

        # corners is a list where each entry corresponds to one detected marker's 4 corners
        for c in corners:
            # Normalize shapes: c may be (4,1,2) or (1,4,2) etc. Reshape to (4,2)
            pts = np.reshape(c, (-1, 2)).astype(np.float32)

            # corner refinement (try; some OpenCV versions expect specific shapes)
            try:
                # cornerSubPix expects image and points in a specific shape: (N,1,2)
                # We'll reshape accordingly for refinement and then reshape back.
                pts_for_cv = pts.reshape(-1, 1, 2)
                cv2.cornerSubPix(gray, pts_for_cv, winSize=(3, 3), zeroZone=(-1, -1), criteria=_CRITERIA)
                pts = pts_for_cv.reshape(-1, 2)
            except Exception:
                # If refinement fails, continue with raw detected corners
                pass

            all_img_pts.append(pts)
            all_obj_pts.append(marker_objp)

        # stack markers for this image into a single view
        imgpoints.append(np.vstack(all_img_pts).astype(np.float32))
        objpoints.append(np.vstack(all_obj_pts).astype(np.float32))
        used_images += 1

    if used_images == 0:
        raise RuntimeError("No markers detected in any images. Cannot calibrate.")

    # Calibrate using the grouped per-image lists
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, last_img_size, None, None
    )
    reproj_error = float(ret) if isinstance(ret, (float, np.floating)) else float(ret[0]) if hasattr(ret, '__len__') else float(ret)
    print(f"Calibrated using {used_images} image(s). Reprojection error: {reproj_error:.6g}")

    return camera_matrix, dist_coeffs, last_img_size, reproj_error


def save_camera_parameters(path: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, image_size: Tuple[int, int], reproj_error: float):
    """
    Saves camera parameters to a .npz file.
    """
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    np.savez_compressed(path,
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                        image_size=np.array(image_size),
                        reprojection_error=reproj_error)
    print(f"Saved camera parameters to: {path}")


def main(argv):
    parser = argparse.ArgumentParser(description="Calibrate cameras using ArUco images (group markers per image).")
    parser.add_argument("--root", default=CALIB_IMG_ROOT, help="Root folder containing calibration subfolders (default from script).")
    parser.add_argument("--prefixes", nargs="+", default=CAMERA_PREFIXES, help="Camera filename prefixes to search for (default from script).")
    parser.add_argument("--out", default=output_file, help="Output file path pattern. Use {prefix} in string to place camera prefix. If not specified, writes <prefix>_cam.npz in current dir.")
    parser.add_argument("--marker-size-mm", type=float, default=size_of_marker, help="Marker size in mm (kept from script).")
    parser.add_argument("--aruco-dict", type=str, default=None, help="Optional ArUco dictionary string (e.g. DICT_4X4_50). If not provided we use the value embedded in the script.")
    args = parser.parse_args(argv)

    root = args.root
    prefixes = args.prefixes
    marker_size_mm = args.marker_size_mm

    # If user passed an explicit aruco-dict name, try to map it
    if args.aruco_dict:
        # Attempt to look up cv2.aruco.<name>
        adname = args.aruco_dict
        try:
            adval = getattr(cv2.aruco, adname)
            aruco_dict_val = adval
        except Exception:
            print(f"Warning: provided aruco dict name '{adname}' not found. Falling back to embedded script value.")
            aruco_dict_val = aruco_dict_type
    else:
        aruco_dict_val = aruco_dict_type

    for prefix in prefixes:
        print(f"\nProcessing camera prefix '{prefix}' under root '{root}' ...")
        image_files = gather_image_paths(root, prefix)
        if len(image_files) == 0:
            print(f"No images found for prefix '{prefix}'. Skipping.")
            continue

        print(f"Found {len(image_files)} image files (first 3 shown): {image_files[:3]}")
        try:
            cam_mtx, dist, img_size, reproj_err = camera_calibration_from_image_list(
                image_files, marker_size_mm, aruco_dict_val
            )
        except Exception as e:
            print(f"Calibration for prefix '{prefix}' failed: {e}")
            continue

        if args.out:
            out_path = args.out.format(prefix=prefix)
        else:
            out_path = f"{prefix}_cam.npz"

        save_camera_parameters(out_path, cam_mtx, dist, img_size, reproj_err)
        print(f"Camera '{prefix}' calibration complete.")
        print(f"Camera Matrix:\n{cam_mtx}")

    print("\nAll done.")


if __name__ == "__main__":
    main(sys.argv[1:])
