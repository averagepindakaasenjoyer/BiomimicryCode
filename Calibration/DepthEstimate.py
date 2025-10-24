import cv2 
import glob

import numpy as np

import os

"""
This module provides functions to estimate depth from stereo images using OpenCV. 
It includes methods for loading stereo image pairs, computing disparity maps, and converting them to depth maps.

Need to change following parameters for different setups:
- file_paths: tuple of paths to stereo image pairs.
- num_disparities: The maximum disparity minus minimum disparity. It must be divisible by 16.
- block_size: The linear size of the blocks compared by the algorithm. It must be an odd number >=1 .
- camera_parameters_path: Path to npz file containing camera parameters.
    - left camera parameters
    - right camera parameters
- stereo_parameters_path: Path to npz file containing stereo calibration parameters.
- target_location: Path to the directory where depth maps will be saved.
"""
file_path = ('/Users/thijnvanveen/Desktop/Biomimicrh/Code/BiomimicryCode/Tezt2/Cam0/camera0pos0.jpg',
             '/Users/thijnvanveen/Desktop/Biomimicrh/Code/BiomimicryCode/Tezt2/Cam1/camera1pos0.jpg') # Paths to calibration images for each camera
num_disparities = 16*5  # Must be divisible by 16 this shows the maximum disparity minus minimum disparity
block_size = 15  # Must be odd and >=1 this is the linear size of the blocks compared by the algorithm.
camera_parameters_path_left = '/Users/thijnvanveen/Desktop/Biomimicrh/Code/BiomimicryCode/calibration_results_cam1.npz'
camera_parameters_path_right = '/Users/thijnvanveen/Desktop/Biomimicrh/Code/BiomimicryCode/calibration_results_cam2.npz'
stereo_parameters_path = '/Users/thijnvanveen/Desktop/Biomimicrh/Code/BiomimicryCode/stereo_calibration_results.npz'
target_location = '/Users/thijnvanveen/Desktop/Biomimicrh/Code/BiomimicryCode/Tezt2/DepthMaps' # Directory to save depth maps

def load_stereo_images(left_image_path, right_image_path):
    """
    Load stereo image pairs from given file paths.
    arguments: left_image_path -- path to the left image
               right_image_path -- path to the right image
    returns: left_image, right_image
    """
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    if left_image is None or right_image is None:
        raise ValueError("Could not load one of the images. Check the file paths.")
    return left_image, right_image

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
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    
    # Compute disparity map
    disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
    
    # Convert disparity to depth
    focal_length = left_camera_matrix[0, 0]  # Assuming fx is at (0,0)
    baseline = np.linalg.norm(stereo_parameters['translation_vector'])
    
    with np.errstate(divide='ignore'):  
        depth_map = (focal_length * baseline) / disparity_map
        depth_map[disparity_map == 0] = 0  # Set depth to 0 where disparity is 0
    
    return depth_map

if __name__ == "__main__":
    # Load camera and stereo parameters
    camera_params_left = np.load(camera_parameters_path_left)
    camera_params_right = np.load(camera_parameters_path_right)
    stereo_params = np.load(stereo_parameters_path)
    print("Camera and stereo parameters loaded.")
    print("Computing depth map.")
    print(camera_parameters_path_left)
    camera_params = (camera_params_left, camera_params_right)
    # load stereo images
    left_image_path = file_path[0]
    right_image_path = file_path[1]
    left_image, right_image = load_stereo_images(left_image_path, right_image_path)
    depth_map = compute_depth_map(left_image, right_image, camera_params, stereo_params)
    
    # Display depth map
    cv2.imshow('Depth Map', depth_map / np.max(depth_map))  # Normalize for display
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    