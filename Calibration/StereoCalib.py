#!/usr/bin/env python3
"""
Batch stereo calibration script.

This script performs stereo calibration for three datasets (8CM, 12CM, 16CM) that live
inside a CalibImg folder sibling to this script's parent directory.

How it works:
- Loads per-camera intrinsics from left_cam_file and right_cam_file (.npz files created by
  your single-camera calibration script).
- For each subfolder (8CM, 12CM, 16CM) it finds matched pairs of images named like
  camera_0.*_<index> and camera_2.*_<index> and runs stereo calibration using detected
  ArUco markers.
- Outputs one stereo calibration .npz per subfolder (e.g. stereo_calibration_results_8cm.npz)

Notes / fixes from your original script:
- Marker side length is taken in millimeters (size_of_marker_mm) and converted to meters
  before use (OpenCV typically expects real-world units to be consistent across object
  points and translation vectors).
- The ArUco detection wrapper is kept for compatibility across OpenCV versions.
- Corner arrays returned from detectMarkers can have slightly different shapes depending
  on OpenCV version; the code normalizes them before calling cornerSubPix.
- If a folder has no matched pairs or no common markers were found, that folder is
  skipped with a warning.

Run:
    python stereo_calibration_batch.py

"""

import os
import glob
import sys
import numpy as np
import cv2

# ----- User config -----
left_cam_file = 'camera_0_cam.npz'   # single-camera calib for left camera
right_cam_file = 'camera_2_cam.npz'  # single-camera calib for right camera

# Root folder that contains subfolders 8CM, 12CM, 16CM. By default it's ../CalibImg
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CALIB_IMG_ROOT = os.path.join(ROOT_DIR, 'CalibImg')
SUBFOLDERS = ['8CM', '12CM', '16CM']

LEFT_PREFIX = 'camera_0'
RIGHT_PREFIX = 'camera_2'

# Marker size in millimeters (convert to meters inside code)
size_of_marker_mm = 26

# ArUco dictionary
aruco_dict_type = cv2.aruco.DICT_4X4_50
<<<<<<< HEAD
=======
max_iterations = 100  # Max iterations for corner refinement
termination_eps = 1e-4  # Desired accuracy for corner refinement
# Create aruco dictionary and parameters
>>>>>>> ab7a01b9e2a8757e2d59232a3adf12b26d9b1d18
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

# Detector parameters (compatibility across OpenCV versions)
try:
    parameters = cv2.aruco.DetectorParameters()
except Exception:
    try:
        parameters = cv2.aruco.DetectorParameters_create()
    except Exception:
        parameters = None

max_iterations = 1000
termination_eps = 1e-9
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, termination_eps)

# ----- Utility functions -----

def detect_markers(gray_img, aruco_dict, parameters=None):
    """Compatibility wrapper for ArUco marker detection.
    Returns (corners, ids, rejected)
    """
    # try classic function
    try:
        func = getattr(cv2.aruco, 'detectMarkers', None)
        if callable(func):
            return func(gray_img, aruco_dict, parameters=parameters)
    except Exception:
        pass

    # newer ArucoDetector class
    try:
        Detector = getattr(cv2.aruco, 'ArucoDetector', None)
        if Detector is not None:
            if parameters is not None:
                detector = Detector(aruco_dict, parameters)
            else:
                detector = Detector(aruco_dict)
            res = detector.detectMarkers(gray_img)
            # res may be (corners, ids, rejected)
            if isinstance(res, tuple) and len(res) >= 3:
                return res[0], res[1], res[2]
            return res
    except Exception:
        pass

    raise RuntimeError("No compatible ArUco detection method available. Ensure opencv-contrib-python is installed.")


def image_paths_from_folder(folder_path, prefix):
    """Return list of files that start with prefix in folder (sorted)."""
    pattern = os.path.join(folder_path, f"{prefix}*")
    files = sorted(glob.glob(pattern))
    return files


# ----- Main stereo calibration function -----

def stereo_vision_calibration(Left_camera_matrix, Left_dist_coeffs, Right_camera_matrix, Right_dist_coeffs, left_images, right_images, marker_size_m):
    """Perform stereo calibration using ArUco single-marker corners from matched image pairs.
    Returns: dict with R, T, Q, E, F, reprojection_error
    """
    if len(left_images) != len(right_images):
        raise ValueError('left_images and right_images must have the same length')

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    # define object points for single marker (4 corners, z=0) in meters
    objp = np.array([[0.0, 0.0, 0.0],
                     [marker_size_m, 0.0, 0.0],
                     [marker_size_m, marker_size_m, 0.0],
                     [0.0, marker_size_m, 0.0]], dtype=np.float32)

    last_image_size = None

    for left_path, right_path in zip(left_images, right_images):
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)
        if img_left is None or img_right is None:
            print(f"Warning: failed to load pair: {left_path}, {right_path}. Skipping.")
            continue

        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        last_image_size = gray_left.shape[::-1]

        corners_left, ids_left, _ = detect_markers(gray_left, aruco_dict, parameters=parameters)
        corners_right, ids_right, _ = detect_markers(gray_right, aruco_dict, parameters=parameters)

        if ids_left is None or ids_right is None:
            continue

        # normalize ids to 1D arrays
        ids_left_arr = np.array(ids_left).reshape(-1)
        ids_right_arr = np.array(ids_right).reshape(-1)

        common_ids = np.intersect1d(ids_left_arr, ids_right_arr)
        if common_ids.size == 0:
            continue

        # corners may come in different shapes depending on OpenCV version; convert to N x 4 x 2
        corners_left_arr = [np.asarray(c).reshape(-1, 2) for c in corners_left]
        corners_right_arr = [np.asarray(c).reshape(-1, 2) for c in corners_right]

        for marker_id in common_ids:
            # find indices of this id in the detected list
            idx_l = int(np.where(ids_left_arr == marker_id)[0][0])
            idx_r = int(np.where(ids_right_arr == marker_id)[0][0])

            cl = corners_left_arr[idx_l].astype(np.float32)
            cr = corners_right_arr[idx_r].astype(np.float32)

            # reshape to shape (4,1,2) for cornerSubPix
            cl_for_refine = cl.reshape(-1, 1, 2)
            cr_for_refine = cr.reshape(-1, 1, 2)

            # refine corners
            try:
                cv2.cornerSubPix(gray_left, cl_for_refine, (3, 3), (-1, -1), criteria)
                cv2.cornerSubPix(gray_right, cr_for_refine, (3, 3), (-1, -1), criteria)
            except Exception:
                # some OpenCV builds may expect different input shapes; ignore refinement failure
                pass

            imgpoints_left.append(cl_for_refine.reshape(-1, 2))
            imgpoints_right.append(cr_for_refine.reshape(-1, 2))
            objpoints.append(objp)

    if len(objpoints) == 0:
        raise ValueError('No common ArUco markers detected in stereo pairs!')

    image_size = last_image_size
    flags = cv2.CALIB_FIX_INTRINSIC

    # stereoCalibrate expects objectPoints as list of arrays of shape (N,3)
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
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

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        Left_camera_matrix, Left_dist_coeffs,
        Right_camera_matrix, Right_dist_coeffs,
        image_size, R, T, flags=0
    )

    return {
        'R': R,
        'T': T,
        'Q': Q,
        'E': E,
        'F': F,
        'R1': R1,
        'R2': R2,
        'P1': P1,
        'P2': P2,
        'roi1': roi1,
        'roi2': roi2,
        'reprojection_error': float(retval)
    }


# ----- Script entry -----
if __name__ == '__main__':
    # check files
    if not os.path.exists(left_cam_file) or not os.path.exists(right_cam_file):
        print(f"Error: expected per-camera calibration files not found: {left_cam_file}, {right_cam_file}")
        sys.exit(1)

    left_data = np.load(left_cam_file)
    right_data = np.load(right_cam_file)

    Left_camera_matrix = left_data['camera_matrix']
    Left_dist_coeffs = left_data['dist_coeffs']
    Right_camera_matrix = right_data['camera_matrix']
    Right_dist_coeffs = right_data['dist_coeffs']

    if not os.path.isdir(CALIB_IMG_ROOT):
        print(f"Error: Calib image root not found: {CALIB_IMG_ROOT}")
        sys.exit(1)

    marker_size_m = float(size_of_marker_mm) / 1000.0

    for sub in SUBFOLDERS:
        folder = os.path.join(CALIB_IMG_ROOT, sub)
        if not os.path.isdir(folder):
            print(f"Warning: subfolder not found, skipping: {folder}")
            continue

        # collect left and right files
        left_files = glob.glob(os.path.join(folder, f"{LEFT_PREFIX}*"))
        right_files = glob.glob(os.path.join(folder, f"{RIGHT_PREFIX}*"))

        def make_map(fl):
            m = {}
            for p in fl:
                key = os.path.basename(p).rsplit('_', 1)[-1]
                m[key] = p
            return m

        lm = make_map(left_files)
        rm = make_map(right_files)

        common_keys = sorted(set(lm.keys()) & set(rm.keys()), key=lambda x: int(x) if x.isdigit() else x)
        if not common_keys:
            print(f"No matched stereo image pairs found in {folder}. Skipping.")
            continue

        matched_left = [lm[k] for k in common_keys]
        matched_right = [rm[k] for k in common_keys]

        print(f"Running stereo calibration for {sub} using {len(matched_left)} matched pairs...")

        try:
            results = stereo_vision_calibration(
                Left_camera_matrix, Left_dist_coeffs,
                Right_camera_matrix, Right_dist_coeffs,
                matched_left, matched_right, marker_size_m
            )
        except Exception as e:
            print(f"Stereo calibration failed for {sub}: {e}")
            continue

        out_filename = f"stereo_calibration_results_{sub.lower()}.npz"
        np.savez(
            out_filename,
            R=results['R'],
            T=results['T'],
            Q=results['Q'],
            E=results['E'],
            F=results['F'],
            R1=results['R1'],
            R2=results['R2'],
            P1=results['P1'],
            P2=results['P2'],
            roi1=results['roi1'],
            roi2=results['roi2'],
            reprojection_error=results['reprojection_error'],
            Left_camera_matrix=Left_camera_matrix,
            Left_dist_coeffs=Left_dist_coeffs,
            Right_camera_matrix=Right_camera_matrix,
            Right_dist_coeffs=Right_dist_coeffs,
            marker_size_m=marker_size_m
        )

        print(f"Saved stereo calibration for {sub} to {out_filename} (reprojection_error={results['reprojection_error']:.6f}).")

    print("All done.")
