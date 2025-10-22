import cv2 
import numpy as np
import glob
import os

"""write the relevant parameters for your ArUco marker detection and calibration here"""
folder_path_left = 'C:\\Users\\Matth\\School2025\\Biomimicry\\BiomimicryCode\\FotosTest\\Cam0'
folder_path_right = 'C:\\Users\\Matth\\School2025\\Biomimicry\\BiomimicryCode\\FotosTest\\Cam1'

# parameters for ArUco marker detection and calibration
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# parameters for the ArUco detector
parameters = cv2.aruco.DetectorParameters()
# size of the ArUco marker in meters 
size_of_marker = 0.04 
# termination criteria (type, maxIterations, Error)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def camera_calibration(images):
    """
    This code will perform camera calibration using ArUco markers.
    It takes in a list of image paths containing ArUco markers and returns the camera matrix and distortion coefficients.
    Code does not handle folders with images, only a list of image paths.
    arguments: images_path -- list of image paths containing ArUco markers
    returns: camera_matrix, dist_coeffs
    """
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # prepare object points based on the size of the marker
    objp = np.array([[0, 0, 0],
                     [size_of_marker, 0, 0],
                     [size_of_marker, size_of_marker, 0],
                     [0, size_of_marker, 0]], dtype=np.float32)
    # Error message in case of no images
    if len(images) == 0:
        raise ValueError("No images provided for calibration.")
    print(f"Starting calibration on {len(images)} images...")
    detected_count = 0
    # loop through all images and detect ArUco markers
    for idx, fname in enumerate(images, start=1):
        print(f"Processing ({idx}/{len(images)}): {fname}")
        # Read the image and convert to grayscale
        img = cv2.imread(fname)
        print(f"  Read image {fname}: {img.shape[1]}x{img.shape[0]} pixels")
        if img is None:
            print(f"  Warning: could not read image {fname}, skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"  Converted to grayscale.")

        # Detect ArUco markers in the image
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        print(f"  Detected markers: {ids.flatten() if ids is not None else []}")

        if ids is not None and len(ids) > 0:
            detected_count += 1
            print(f"  Found {len(ids)} markers in {fname}.")
            for corner in corners:
                # search for subpixel corners in the detected corners
                try:
                    cv2.cornerSubPix(gray, corner, winSize=(3,3), zeroZone=(-1,-1), criteria=criteria)
                except Exception:
                    # some OpenCV versions expect different input shapes; ignore failure here
                    pass
                imgpoints.append(corner.reshape(-1, 2))
                objpoints.append(objp)
        else:
            print(f"  No markers detected in {fname}.")
    
    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise ValueError("No ArUco markers were detected in any of the provided images. Calibration cannot proceed.")

    # Perform camera calibration to get camera matrix and distortion coefficients
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(f"Calibration finished: used {len(objpoints)} views, reproj error={ret}")
    return camera_matrix, dist_coeffs

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


def stereo_vision_calibration(Left_camera_matrix, Left_dist_coeffs, Right_camera_matrix, Right_dist_coeffs):
    """
    Perform stereo calibration and rectification using two calibrated cameras.
    This assumes both cameras have been calibrated individually (intrinsics known)
    and you have synchronized images of the same ArUco board or pattern.
    """

    # Load image paths from both folders using the helper so multiple extensions are supported
    left_images = image_paths_from_folder(folder_path_left)
    right_images = image_paths_from_folder(folder_path_right)

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

def compute_depth_map(left_image_path, right_image_path,
                      Left_camera_matrix, Left_dist_coeffs,
                      Right_camera_matrix, Right_dist_coeffs,
                      R, T, Q):
    """
    Compute a depth map from a pair of stereo images and known calibration parameters.

    Arguments:
        left_image_path, right_image_path : str
            File paths to the left and right images.
        Left_camera_matrix, Left_dist_coeffs : np.ndarray
            Intrinsic parameters of the left camera.
        Right_camera_matrix, Right_dist_coeffs : np.ndarray
            Intrinsic parameters of the right camera.
        R, T : np.ndarray
            Rotation and translation between cameras (from stereo calibration).
        Q : np.ndarray
            Disparity-to-depth mapping matrix (from stereoRectify).
    Returns:
        disparity : np.ndarray
            Computed disparity map.
        depth_map : np.ndarray
            Depth values (in the same units as your marker size, typically meters).
    """

    # === Load and prepare images ===
    img_left = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
    img_right = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # === Rectify and undistort ===
    image_size = gray_left.shape[::-1]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        Left_camera_matrix, Left_dist_coeffs,
        Right_camera_matrix, Right_dist_coeffs,
        image_size, R, T, flags=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        Left_camera_matrix, Left_dist_coeffs, R1, P1, image_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        Right_camera_matrix, Right_dist_coeffs, R2, P2, image_size, cv2.CV_32FC1
    )

    rectified_left = cv2.remap(gray_left, map1x, map1y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(gray_right, map2x, map2y, cv2.INTER_LINEAR)

    # === Compute disparity ===
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=96,   # Must be divisible by 16
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    disparity = stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0

    # === Convert disparity to depth ===
    depth_map = cv2.reprojectImageTo3D(disparity, Q)

    # Optional: Normalize for visualization
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    cv2.imshow("Rectified Left", rectified_left)
    cv2.imshow("Rectified Right", rectified_right)
    cv2.imshow("Disparity", disp_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return disparity, depth_map


if __name__ == "__main__":
    # Automatically detect common image extensions in the folders
    image_paths_left = image_paths_from_folder(folder_path_left)
    print("Found", len(image_paths_left), "images in left folder for calibration.")
    if len(image_paths_left) == 0:
        print(f"No images found in left folder '{folder_path_left}'. Check the path and file extensions.")
        raise SystemExit(1)
    camera_matrix_left, dist_coeffs_left = camera_calibration(image_paths_left)
    image_paths_right = image_paths_from_folder(folder_path_right)
    print("Found", len(image_paths_right), "images in right folder for calibration.")
    if len(image_paths_right) == 0:
        print(f"No images found in right folder '{folder_path_right}'. Check the path and file extensions.")
        raise SystemExit(1)
    camera_matrix_right, dist_coeffs_right = camera_calibration(image_paths_right)

    print("Left Camera Matrix:\n", camera_matrix_left)
    print("Right Camera Matrix:\n", camera_matrix_right)

    R, T, Q = stereo_vision_calibration(camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right)

    #test images
    test_left = image_paths_left[0]
    test_right = image_paths_right[0]

    # calculate depth map and use it to get distance to objects
    disparity, depth_map = compute_depth_map(
        test_left, test_right,
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        R, T, Q
    )

    print("Example depth at center pixel:", depth_map[depth_map.shape[0]//2, depth_map.shape[1]//2])
    print("Example depth at (100, 100):", depth_map[100, 100], " (X, Y, Z in meters)")

    