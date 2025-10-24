import cv2 
import numpy as np
import glob
import os
"""
This file will contain camera calibration functions.
This will return a camara matrices for a given set of camaras, using aruco markers.

Need to change following prameters for different setups:
- file_paths: List of paths to calibration images for each camera. Order matters and should correspond to camera numbers.
- size_of_marker: The length of the aruco marker's side in meters.
- output_file: The list names for the files where calibration results will be saved.
    - default: 'calibration_results_camX.npz' with X being the camera number.
- aruco_dict_type: The type of aruco dictionary used for marker detection.
    - default: cv2.aruco.DICT_4X4_50 This is the aruco board available from the computer vision lab.
- max_iterations: The maximum number of iterations for corner refinement.
- termination_eps: The desired accuracy for corner refinement. (this is between consecutive iterations shows convergences)
"""
file_path = ['/Users/thijnvanveen/Desktop/Biomimicrh/Code/BiomimicryCode/Tezt2/Cam0',
             '/Users/thijnvanveen/Desktop/Biomimicrh/Code/BiomimicryCode/Tezt2/Cam1'] # Paths to calibration images for each camera
size_of_marker = 0.04  # Marker side length in meters
output_file = ['calibration_results_cam1.npz', 'calibration_results_cam2.npz'] # Output file names
aruco_dict_type = cv2.aruco.DICT_4X4_50 # Aruco dictionary type
max_iterations = 1000  # Max iterations for corner refinement
termination_eps = 1e-9  # Desired accuracy for corner refinement

# Create aruco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
parameters = cv2.aruco.DetectorParameters()
# Define criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, termination_eps)


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
        raise ValueError("No images provided for calibration. Most likely the image paths are incorrect or the folder is empty.")
    print(f"Starting calibration on {len(images)} images.")
    detected_count = 0
    # loop through all images and detect ArUco markers
    for idx, fname in enumerate(images, start=1):
        print(f"Processing ({idx}/{len(images)}): {fname}")
        # Read the image and convert to grayscale
        img = cv2.imread(fname)
        print(f"  Read image {fname}: {img.shape[1]}x{img.shape[0]} pixels")
        if img is None:
            print(f"  Warning: could not read image {fname}, skipping. Make sure the file exists and is an image.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"  Converted the image to grayscale.")

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

if __name__ == "__main__":
    # This loop will calibrate all cameras based on the output_file list
    for i in range(len(output_file)):
        print(f"Calibrating camera {i+1}/{len(output_file)} using images from: {file_path[i]}")
        images = image_paths_from_folder(file_path[i])
        camera_matrix, dist_coeffs = camera_calibration(images)
        # Save the calibration results
        np.savez(output_file[i], camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print(f"Saved calibration results to {output_file[i]}\n")


