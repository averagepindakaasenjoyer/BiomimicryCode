import cv2 
import glob
import numpy as np
import os
import time
import dotenv

dotenv.load_dotenv()

"""

This module provides functions to estimate depth from live camera feeds using stereo vision techniques. 
It captures frames from two cameras, computes disparity maps, and converts them to depth maps in real-time.
Need to change following parameters for different setups:
- num_disparities: The maximum disparity minus minimum disparity. It must be divisible by 16.
- block_size: The linear size of the blocks compared by the algorithm. It must be an odd number >=1 .
- camera_parameters_path: Path to npz file containing camera parameters.
    - left camera parameters
    - right camera parameters
- stereo_parameters_path: Path to npz file containing stereo calibration parameters.
- show_depth_map: Boolean flag to display the computed depth map.
"""
num_disparities = 16 * 5  # Must be divisible by 16
block_size = 15  # Must be odd and >=1
camera_parameters_path_left = os.getenv("LEFT_CAM_PARAMS")
camera_parameters_path_right = os.getenv("RIGHT_CAM_PARAMS")
stereo_parameters_path = os.getenv("STEREO_PARAMS")
target_location = os.getenv("TARGET_LOCATION_DEPTHMAPS")  # Directory to save depth maps
show_depth_map = False  # Flag to display depth map
pixel_to_determine = (320, 240)  # Pixel coordinates to determine depth (x, y)

def live_camera_data(camera_index_left, camera_index_right):
    """
    Capture live video from two cameras and prepare frames for depth estimation.
    returns: left_frame, right_frame
    """
    # Open video capture for left and right cameras (camera indices may need to be adjusted)
    cap_left = cv2.VideoCapture(camera_index_left)
    cap_right = cv2.VideoCapture(camera_index_right)

    if not cap_left.isOpened() or not cap_right.isOpened():
        raise ValueError("Could not open one of the camera streams. Check camera connections and indices.")

    ret_left, left_frame = cap_left.read()
    ret_right, right_frame = cap_right.read()
    if not ret_left or not ret_right:
        raise ValueError("Could not read frames from one of the cameras.")
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)

    if not cap_left.isOpened() or not cap_right.isOpened():
        raise ValueError("Could not open one of the camera streams. Check camera connections and indices.")
    
    ret_left, left_frame = cap_left.read()
    ret_right, right_frame = cap_right.read()
    if not ret_left or not ret_right:
        raise ValueError("Could not read frames from one of the cameras.")
    
    # make gray scale
    left_frame_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_frame_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    # Release the video captures
    cap_left.release()
    cap_right.release()
    return left_frame_gray, right_frame_gray


def compute_depth_map(left_image, right_image, camera_parameters, stereo_parameters):
    """
    Compute depth map from stereo images.
    arguments: left_image -- left image of the stereo pair
               right_image -- right image of the stereo pair
               camera_parameters -- dictionary containing camera matrices and distortion coefficients
               stereo_parameters -- dictionary containing stereo calibration parameters
    returns: depth_map
    """
    left_camera_matrix = camera_parameters[0]
    right_camera_matrix = camera_parameters[1]

    # Create StereoBM object
    stereo = cv2.StereoSGBM_create(numDisparities=num_disparities, blockSize=block_size)

    # Compute disparity map
    disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16.0

    # Convert disparity to depth
    focal_length = left_camera_matrix[0, 0]  # Assuming fx is at (0,0)
    baseline = np.linalg.norm(stereo_parameters['T'])

    with np.errstate(divide='ignore'):
        depth_map = (focal_length * baseline) / disparity_map
        depth_map[disparity_map == 0] = 0

    return depth_map

def check_camera_indices():
    """
    This function loops over all possible indices for the cameras and returns the first two that are found.
    This way the code works even if the camera indices are not 0 and 1.
    returns: left_camera_index, right_camera_index
    """
    left_camera_index = None
    right_camera_index = None
    for i in range(10): 
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if left_camera_index is None:
                    left_camera_index = i
                else:
                    right_camera_index = i
                    break
        cap.release()
    if left_camera_index is None or right_camera_index is None:
        raise ValueError("No two cameras found, check camera connections.")
    return left_camera_index, right_camera_index

if __name__ == "__main__":
    left_camera_index, right_camera_index = check_camera_indices()
    print(f"Using camera indices - Left: {left_camera_index}, Right: {right_camera_index}")

    # Load camera and stereo parameters
    camera_params_left = np.load(camera_parameters_path_left)
    camera_params_right = np.load(camera_parameters_path_right)
    stereo_params = np.load(stereo_parameters_path)

    left_camera_matrix = camera_params_left['camera_matrix']
    right_camera_matrix = camera_params_right['camera_matrix']
    translation_vector = stereo_params['T']

    camera_params = (left_camera_matrix, right_camera_matrix)
    stereo_params = {'T': translation_vector}

    print("Camera and stereo parameters loaded.")
    print("Computing depth map.")
    print(camera_parameters_path_left)
    start_time = time.time()
    left_frame, right_frame = live_camera_data(left_camera_index, right_camera_index)
    depth_map = compute_depth_map(left_frame, right_frame, camera_params, stereo_params)


    # display depth to center pixel
    center_x, center_y = pixel_to_determine
    print(f'Depth at center pixel ({center_x}, {center_y}): {depth_map[center_y, center_x]:.2f} meters')
    time_elapsed = time.time() - start_time
    print(f'Computed depth to center in {time_elapsed:.2f} seconds.')
