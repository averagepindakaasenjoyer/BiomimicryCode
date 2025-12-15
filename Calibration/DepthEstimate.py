import cv2 
import glob
import numpy as np
import os
import time
import dotenv

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
- show_depth_map: Boolean flag to display the computed depth map.
"""

dotenv.load_dotenv()

Calibration_Img_Dir = os.getenv('CALIB_IMGS_DIR')


Distance_Cams_Folders = ['8CM_converted', '12CM_converted', '16CM_converted']

Img_Couples = {
                '0': {'left': 'camera_0_0.jpg', 'right': 'camera_2_0.jpg'},
                '1': {'left': 'camera_0_1.jpg', 'right': 'camera_2_1.jpg'},
                '2': {'left': 'camera_0_2.jpg', 'right': 'camera_2_2.jpg'},
                '3': {'left': 'camera_0_3.jpg', 'right': 'camera_2_3.jpg'},
                '4': {'left': 'camera_0_4.jpg', 'right': 'camera_2_4.jpg'},
                '5': {'left': 'camera_0_5.jpg', 'right': 'camera_2_5.jpg'}, 
                '6': {'left': 'camera_0_6.jpg', 'right': 'camera_2_6.jpg'},
                '7': {'left': 'camera_0_7.jpg', 'right': 'camera_2_7.jpg'},
                '8': {'left': 'camera_0_8.jpg', 'right': 'camera_2_8.jpg'},
                '9': {'left': 'camera_0_9.jpg', 'right': 'camera_2_9.jpg'},
                '10': {'left': 'camera_0_10.jpg', 'right': 'camera_2_10.jpg'},
                '11': {'left': 'camera_0_11.jpg', 'right': 'camera_2_11.jpg'},
                '12': {'left': 'camera_0_12.jpg', 'right': 'camera_2_12.jpg'},
                '13': {'left': 'camera_0_13.jpg', 'right': 'camera_2_13.jpg'},
                '14': {'left': 'camera_0_14.jpg', 'right': 'camera_2_14.jpg'},
                '15': {'left': 'camera_0_15.jpg', 'right': 'camera_2_15.jpg'},
                '16': {'left': 'camera_0_16.jpg', 'right': 'camera_2_16.jpg'},
                '17': {'left': 'camera_0_17.jpg', 'right': 'camera_2_17.jpg'}
            }

SELECTED_IMG_COUPLE = '5'  # Change this index to select different image pairs
SELECTED_FOLDER_INDEX = 0  # 0 for 8CM, 1 for 12CM, 2 for 16CM
num_disparities = 16 * 5  # Must be divisible by 16
block_size = 15  # Must be odd and >=1

file_path = (os.path.join(Calibration_Img_Dir, Distance_Cams_Folders[SELECTED_FOLDER_INDEX], Img_Couples[SELECTED_IMG_COUPLE]['left']),
             os.path.join(Calibration_Img_Dir, Distance_Cams_Folders[SELECTED_FOLDER_INDEX], Img_Couples[SELECTED_IMG_COUPLE]['right']))

camera_parameters_path_left = os.getenv('LEFT_CAM_PARAMS')
camera_parameters_path_right = os.getenv('RIGHT_CAM_PARAMS')


if SELECTED_FOLDER_INDEX == 0:
    stereo_parameters_path = os.getenv('STEREO_PARAMS_8CM')
elif SELECTED_FOLDER_INDEX == 1:
    stereo_parameters_path = os.getenv('STEREO_PARAMS_12CM')
else:
    stereo_parameters_path = os.getenv('STEREO_PARAMS_16CM')

target_location = os.getenv('TARGET_LOCATION_DEPTHMAPS', '/')  # Directory to save depth maps
show_depth_map = False  # Flag to display depth map


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
    Compute depth map from stereo images using rectification.
    arguments: left_image -- left image of the stereo pair
               right_image -- right image of the stereo pair
               camera_parameters -- tuple of (left_camera_matrix, right_camera_matrix, left_dist_coeffs, right_dist_coeffs)
               stereo_parameters -- dictionary containing stereo calibration parameters
    returns: depth_map
    """
    left_camera_matrix = camera_parameters[0]
    right_camera_matrix = camera_parameters[1]
    left_dist_coeffs = camera_parameters[2]
    right_dist_coeffs = camera_parameters[3]

    # Get stereo rectification parameters
    R1 = stereo_parameters['R1']
    R2 = stereo_parameters['R2']
    P1 = stereo_parameters['P1']
    P2 = stereo_parameters['P2']
    Q = stereo_parameters['Q']
    T = stereo_parameters['T']
    print(f"Stereo baseline (T): {T}")
    print("rotation matrix R1:\n", R1)
    print("projection matrix P1:\n", P1)
    print("rotation matrix R2:\n", R2)
    print("projection matrix P2:\n", P2)
    print("disparity-to-depth mapping matrix Q:\n", Q)
    T[0,0] = -16
    print(f"Stereo baseline (T): {T}")

    
    image_size = left_image.shape[::-1]  # (width, height)
    
    # Compute rectification maps
    mapL1, mapL2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_dist_coeffs, R1, P1, image_size, cv2.CV_32F)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_dist_coeffs, R2, P2, image_size, cv2.CV_32F)
    
    # Apply rectification
    left_rectified = cv2.remap(left_image, mapL1, mapL2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_image, mapR1, mapR2, cv2.INTER_LINEAR)
    
    # Use StereoSGBM for potentially better matching than StereoBM
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Compute disparity map on rectified images
    disparity_map = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0

    
    # Debug: Check disparity range
    valid_disparities = disparity_map[disparity_map > 0]
    if len(valid_disparities) > 0:
        print(f"Disparity range: {valid_disparities.min():.2f} to {valid_disparities.max():.2f}")
    else:
        print("Warning: No valid disparities found!")
    
    # Convert disparity to depth using the Q matrix
    # The Q matrix encodes: Z = -f*B / disparity where f is focal length and B is baseline
    # The formula is: depth = Q[2,3] / disparity (note: Q[2,3] is typically negative)
    
    depth_map = np.zeros_like(disparity_map)
    valid_mask = disparity_map > 0  # Only positive disparities yield valid depth
    
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     # Q[2,3] encodes -f*baseline, so division gives depth directly
    #     depth_map[valid_mask] = Q[2, 3] / disparity_map[valid_mask]
    # after you have disparity_map (in pixels, i.e. after /16)
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q)   # returns X,Y,Z in units of T (units of baseline)
    depth_map = points_3d[:, :, 2]                         # Z channel

    # mask invalid disparities
    invalid_mask = (disparity_map <= 0) | ~np.isfinite(depth_map)
    depth_map[invalid_mask] = 0.0
    # Ensure positive depths (take absolute value)
    depth_map = np.abs(depth_map)
    
    return depth_map


if __name__ == "__main__":
    start_time = time.time()
    # Load camera and stereo parameters
    camera_params_left = np.load(camera_parameters_path_left)
    camera_params_right = np.load(camera_parameters_path_right)
    stereo_params = np.load(stereo_parameters_path)

    left_camera_matrix = camera_params_left['camera_matrix']
    left_dist_coeffs = camera_params_left['dist_coeffs']
    right_camera_matrix = camera_params_right['camera_matrix']
    right_dist_coeffs = camera_params_right['dist_coeffs']

    camera_params = (left_camera_matrix, right_camera_matrix, left_dist_coeffs, right_dist_coeffs)
    stereo_params = {
        'T': stereo_params['T'],
        'R1': stereo_params['R1'],
        'R2': stereo_params['R2'],
        'P1': stereo_params['P1'],
        'P2': stereo_params['P2'],
        'Q': stereo_params['Q']
    }

    print("Camera and stereo parameters loaded.")
    print("Computing depth map.")
    print(camera_parameters_path_left)

    # Load stereo images
    left_image_path = file_path[0]
    right_image_path = file_path[1]
    left_image, right_image = load_stereo_images(left_image_path, right_image_path)

    depth_map = compute_depth_map(left_image, right_image, camera_params, stereo_params)

    # Display depth map
    print('Depth map computed. Displaying result in a window.')
    print("distance to center pixel (in meters): ", depth_map[depth_map.shape[0]//2, depth_map.shape[1]//2])
    depth_map_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if show_depth_map:
        cv2.imshow('Depth Map', depth_map_display)
        cv2.waitKey(0)
    # Save depth map
    time_elapsed = time.time() - start_time
    print(f'Depth map computation completed in {time_elapsed:.2f} seconds.')
    cv2.imwrite(os.path.join(target_location, 'depth_map.png'), depth_map_display)  
    print(f'Depth map saved to {target_location}')
    cv2.destroyAllWindows()
