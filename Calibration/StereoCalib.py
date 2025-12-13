"""
Batch stereo calibration script (Windows-safe, no multiprocessing).

The file assumes that individual camera intrinsics have already been
calibrated and saved in 'camera_0_cam.npz' and 'camera_2_cam.npz'.
It processes image pairs from subfolders under 'CalibImg' and looks for
images starting with 'camera_0' and 'camera_2' respectively.
Saves stereo calibration results in separate .npz files per subfolder.

This script can take a few hours to run depending on the number of image pairs.
"""

import os
import glob
import sys
import numpy as np
import cv2
from tqdm import tqdm


# USER CONFIG
left_cam_file = 'camera_0_cam.npz'
right_cam_file = 'camera_2_cam.npz'

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CALIB_IMG_ROOT = os.path.join(ROOT_DIR, 'CalibImg')
SUBFOLDERS = [
    #'8CM',
    #'12CM',
    '16CM'
    ]

LEFT_PREFIX = 'camera_0'
RIGHT_PREFIX = 'camera_2'

size_of_marker_mm = 26
aruco_dict_type = cv2.aruco.DICT_4X4_50


# ARUCO SETUP, sometimes version-dependent
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
try:
    parameters = cv2.aruco.DetectorParameters()
except Exception:
    parameters = cv2.aruco.DetectorParameters_create()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-4)


# ARUCO DETECTION WRAPPER
def detect_markers(gray, dictionary, parameters):
    """Version-agnostic ArUco detection."""
    # Older OpenCV API
    try:
        func = getattr(cv2.aruco, 'detectMarkers', None)
        if callable(func):
            return func(gray, dictionary, parameters=parameters)
    except:
        pass
    try:
        Detector = getattr(cv2.aruco, 'ArucoDetector', None)
        if Detector:
            det = Detector(dictionary, parameters)
            return det.detectMarkers(gray)
    except:
        pass

    raise RuntimeError("No ArUco detection method available.")


# STEREO CALIBRATION
def stereo_vision_calibration(
        Left_camera_matrix, Left_dist_coeffs,
        Right_camera_matrix, Right_dist_coeffs,
        left_images, right_images, marker_size_m):

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    # A 4-corner square as object points
    objp = np.array([
        [0, 0, 0],
        [marker_size_m, 0, 0],
        [marker_size_m, marker_size_m, 0],
        [0, marker_size_m, 0]
    ], dtype=np.float32)

    last_image_size = None

    # Detect corners for all image pairs
    for left_path, right_path in tqdm(
        list(zip(left_images, right_images)),
        desc="Extracting markers",
        unit="pair"
    ):
        imgL = cv2.imread(left_path)
        imgR = cv2.imread(right_path)
        if imgL is None or imgR is None:
            continue

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        last_image_size = grayL.shape[::-1]

        cornersL, idsL, _ = detect_markers(grayL, aruco_dict, parameters)
        cornersR, idsR, _ = detect_markers(grayR, aruco_dict, parameters)

        if idsL is None or idsR is None:
            continue

        idsL = np.array(idsL).reshape(-1)
        idsR = np.array(idsR).reshape(-1)
        common = np.intersect1d(idsL, idsR)

        if len(common) == 0:
            continue

        cL = [np.array(c).reshape(-1,2) for c in cornersL]
        cR = [np.array(c).reshape(-1,2) for c in cornersR]

        for mid in common:
            iL = np.where(idsL == mid)[0][0]
            iR = np.where(idsR == mid)[0][0]

            pL = cL[iL].astype(np.float32).reshape(-1,1,2)
            pR = cR[iR].astype(np.float32).reshape(-1,1,2)

            try:
                cv2.cornerSubPix(grayL, pL, (3,3), (-1,-1), criteria)
                cv2.cornerSubPix(grayR, pR, (3,3), (-1,-1), criteria)
            except:
                pass

            imgpoints_left.append(pL.reshape(-1,2))
            imgpoints_right.append(pR.reshape(-1,2))
            objpoints.append(objp)

    if len(objpoints) == 0:
        raise ValueError("No usable marker correspondences found.")

    image_size = last_image_size

    # Stereo calibration (no timeout) because it can take a while
    retval, CM1, DC1, CM2, DC2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        Left_camera_matrix,
        Left_dist_coeffs,
        Right_camera_matrix,
        Right_dist_coeffs,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        Left_camera_matrix, Left_dist_coeffs,
        Right_camera_matrix, Right_dist_coeffs,
        image_size, R, T, flags=0
    )

    return dict(
        R=R, T=T, Q=Q, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2,
        roi1=roi1, roi2=roi2,
        reprojection_error=float(retval)
    )

if __name__ == "__main__":
    if not os.path.exists(left_cam_file) or not os.path.exists(right_cam_file):
        print("Missing left or right camera intrinsics.")
        sys.exit(1)

    left_data = np.load(left_cam_file)
    right_data = np.load(right_cam_file)

    LCM = left_data["camera_matrix"]
    LDC = left_data["dist_coeffs"]
    RCM = right_data["camera_matrix"]
    RDC = right_data["dist_coeffs"]

    marker_size_m = size_of_marker_mm / 1000.0

    for sub in SUBFOLDERS:
        folder = os.path.join(CALIB_IMG_ROOT, sub)
        if not os.path.isdir(folder):
            print(f"Missing folder {folder}, skipping.")
            continue

        left_files = sorted(glob.glob(os.path.join(folder, f"{LEFT_PREFIX}*")))
        right_files = sorted(glob.glob(os.path.join(folder, f"{RIGHT_PREFIX}*")))

        # Index-matching
        def map_idx(files):
            m = {}
            for p in files:
                key = os.path.basename(p).rsplit('_', 1)[-1]
                m[key] = p
            return m

        lm = map_idx(left_files)
        rm = map_idx(right_files)

        common = sorted(
            set(lm.keys()) & set(rm.keys()),
            key=lambda x: int(x) if x.isdigit() else x
        )

        if not common:
            print(f"No matched pairs in {folder}.")
            continue

        left_list = [lm[k] for k in common]
        right_list = [rm[k] for k in common]

        print(f"\n=== Processing {sub}: {len(left_list)} pairs ===")

        try:
            results = stereo_vision_calibration(
                LCM, LDC, RCM, RDC,
                left_list, right_list, marker_size_m
            )
        except Exception as e:
            print(f"❌ Error in {sub}: {e}")
            continue

        outname = f"stereo_calibration_results_{sub.lower()}.npz"
        np.savez(
            outname,
            **results,
            Left_camera_matrix=LCM,
            Left_dist_coeffs=LDC,
            Right_camera_matrix=RCM,
            Right_dist_coeffs=RDC,
            marker_size_m=marker_size_m
        )

        print(f"✔ Saved {outname}  (reproj error={results['reprojection_error']:.6f})")

    print("All done.")
