import cv2 
import numpy as np
import glob
import os

"""
This file will perform stereo camera calibration using ArUco markers and the camera matrices.
It assumes that individual camera calibrations have already been performed using CamaraCalib.py and that the calibration results are saved in .npz files.

Need to change following prameters for different setups:
- left_cam_file: The file path to the left camera's calibration results (.npz file)
- right_cam_file: The file path to the right camera's calibration results (.npz file)
- stereo_output_file: The file name for saving the stereo calibration results.
    - default: 'stereo_calibration_results.npz'
- file_path: List of paths to calibration images for each camera. Order matters and should correspond to camera numbers.
- size_of_marker: The length of the aruco marker's side in meters.
- aruco_dict_type: The type of aruco dictionary used for marker detection.
    - default: cv2.aruco.DICT_4X4_50 This is the aruco board available from the makerspace Aruco board.
- max_iterations: The maximum number of iterations for corner refinement.
- termination_eps: The desired accuracy for corner refinement. (this is between consecutive iterations shows convergences)

"""
left_cam_file = 'calibration_results_camera_0.npz'  # Left camera calibration results (produced by CamaraCalib)
right_cam_file = 'calibration_results_camera_2.npz'  # Right camera calibration results (produced by CamaraCalib)
stereo_output_file = 'stereo_calibration_results.npz'  # Output file for stereo calibration results
# Use repository-level CalibImg folder by default
CALIB_IMG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CalibImg'))
LEFT_PREFIX = 'camera_0'
RIGHT_PREFIX = 'camera_2'
size_of_marker = 26  # Marker side length in mm
aruco_dict_type = cv2.aruco.DICT_4X4_50
max_iterations = 1000  # Max iterations for corner refinement
termination_eps = 1e-9  # Desired accuracy for corner refinement
# Create aruco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
parameters = cv2.aruco.DetectorParameters()
# Define criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, termination_eps)

def image_paths_from_folder(folder_path, extensions=None):
    """
    Retrieve image paths from a folder. Searches a set of common extensions when
    `extensions` is None. Returns a sorted list.

    arguments:
        folder_path -- path to the folder containing images
        extensions -- list of extensions (e.g. ['jpg','jpeg']) or None to use common ones
    returns: list of image paths
    """
    if extensions is None:
        extensions = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']

    image_paths = []
    for ext in extensions:
        pattern = os.path.join(folder_path, f"*.{ext}")
        image_paths.extend(glob.glob(pattern))

    # remove duplicates and sort for deterministic order
    image_paths = sorted(list(dict.fromkeys(image_paths)))
    return image_paths

def stereo_vision_calibration(Left_camera_matrix, Left_dist_coeffs, Right_camera_matrix, Right_dist_coeffs, left_images, right_images):
    """
    Perform stereo calibration and rectification using two calibrated cameras.
    This assumes both cameras have been calibrated individually (intrinsics known)
    and you have synchronized images of the same ArUco board or pattern.
    arguments:
        Left_camera_matrix -- Intrinsic matrix of the left camera
        Left_dist_coeffs -- Distortion coefficients of the left camera
        Right_camera_matrix -- Intrinsic matrix of the right camera
        Right_dist_coeffs -- Distortion coefficients of the right camera
        folder_path_left -- Path to folder with left camera images
        folder_path_right -- Path to folder with right camera images
    returns: R, T, Q matrices from stereo calibration
    """

    # left_images and right_images should be lists of equal-length matched paths
    # Prepare ArUco detector
    objpoints = []  # 3D points in real world
    imgpoints_left = []
    imgpoints_right = []

    # Define marker coordinates (same as before)
    objp = np.array([[0, 0, 0],
                     [size_of_marker, 0, 0],
                     [size_of_marker, size_of_marker, 0],
                     [0, size_of_marker, 0]], dtype=np.float32)

    # Detect ArUco markers in both image sets
    for left_path, right_path in zip(left_images, right_images):
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)
        if img_left is None or img_right is None:
            # skip if either image failed to load
            continue
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # detect markers
        corners_left, ids_left, _ = cv2.aruco.detectMarkers(gray_left, aruco_dict, parameters=parameters)
        corners_right, ids_right, _ = cv2.aruco.detectMarkers(gray_right, aruco_dict, parameters=parameters)

        if ids_left is not None and ids_right is not None:
            # Make sure same markers are detected in both
            common_ids = np.intersect1d(ids_left.flatten(), ids_right.flatten())
            for marker_id in common_ids:
                idx_l = np.where(ids_left == marker_id)[0][0]
                idx_r = np.where(ids_right == marker_id)[0][0]
                cv2.cornerSubPix(gray_left, corners_left[idx_l], (3,3), (-1,-1), criteria)
                cv2.cornerSubPix(gray_right, corners_right[idx_r], (3,3), (-1,-1), criteria)
                imgpoints_left.append(corners_left[idx_l].reshape(-1, 2))
                imgpoints_right.append(corners_right[idx_r].reshape(-1, 2))
                objpoints.append(objp)

    if len(objpoints) == 0:
        raise ValueError("No common ArUco markers detected in stereo pairs!")

    image_size = gray_left.shape[::-1]

    # Use the cv2 stereoCalibrate function to find R and T between the two cameras
    flags = cv2.CALIB_FIX_INTRINSIC
    retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        Left_camera_matrix,
        Left_dist_coeffs,
        Right_camera_matrix,
        Right_dist_coeffs,
        image_size,
        criteria=criteria,
        flags=flags
    )

    # Stereo rectification to compute Q matrix
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        Left_camera_matrix, Left_dist_coeffs,
        Right_camera_matrix, Right_dist_coeffs,
        image_size, R, T, flags=0
    )
    
    return R, T, Q

if __name__ == "__main__":
    # Load individual camera calibration results produced by `CamaraCalib.py`.
    if not os.path.exists(left_cam_file) or not os.path.exists(right_cam_file):
        raise SystemExit(f"Expected per-camera calibration files not found: {left_cam_file}, {right_cam_file}")
    left_data = np.load(left_cam_file)
    right_data = np.load(right_cam_file)

    Left_camera_matrix = left_data['camera_matrix']
    Left_dist_coeffs = left_data['dist_coeffs']
    Right_camera_matrix = right_data['camera_matrix']
    Right_dist_coeffs = right_data['dist_coeffs']

    # Build matched pairs across all subfolders in CALIB_IMG_ROOT by matching trailing indices
    if not os.path.isdir(CALIB_IMG_ROOT):
        raise SystemExit(f"Calib image root not found: {CALIB_IMG_ROOT}")

    matched_left = []
    matched_right = []
    for sub in sorted(os.listdir(CALIB_IMG_ROOT)):
        subp = os.path.join(CALIB_IMG_ROOT, sub)
        if not os.path.isdir(subp):
            continue
        left_files = glob.glob(os.path.join(subp, f"{LEFT_PREFIX}*"))
        right_files = glob.glob(os.path.join(subp, f"{RIGHT_PREFIX}*"))
        # map by trailing index after last underscore e.g. camera_0.jpg_3 -> '3'
        def make_map(fl):
            m = {}
            for p in fl:
                key = os.path.basename(p).rsplit('_', 1)[-1]
                m[key] = p
            return m

        lm = make_map(left_files)
        rm = make_map(right_files)
        for k in sorted(set(lm.keys()) & set(rm.keys()), key=lambda x: int(x) if x.isdigit() else x):
            matched_left.append(lm[k])
            matched_right.append(rm[k])

    if len(matched_left) == 0:
        raise SystemExit(f"No matched stereo image pairs found under {CALIB_IMG_ROOT}")

    print(f"Starting stereo calibration using {len(matched_left)} matched pairs from {CALIB_IMG_ROOT}.")
    R, T, Q = stereo_vision_calibration(
        Left_camera_matrix, Left_dist_coeffs,
        Right_camera_matrix, Right_dist_coeffs,
        matched_left, matched_right
    )

    # Save stereo calibration results
    np.savez(stereo_output_file, R=R, T=T, Q=Q)
    print(f"Stereo calibration results saved to {stereo_output_file}.")