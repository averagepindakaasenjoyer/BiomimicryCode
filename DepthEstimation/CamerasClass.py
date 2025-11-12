"""
This file will contain the Cameras class for depth estimation tasks.
Structure:
- Class: Cameras
    - Sub-classes:
        - camera
            - intrinsic_parameters
            - camera_index
            - image_width
            - image_height
        - depth_estimation
            - stereo_parameters
            functions:
                - compute_depth_map()
                - compute_depth_at_point()
        - detection
            - object_detection_parameters
            functions:
                - detect_objects_in_image()
                - get_bounding_boxes()
"""
import cv2 
import numpy as np
import glob
import os


class Cameras:
    def __init__(self):
        self.camera = self.camera()
        self.depth_estimation = self.depth_estimation()
        self.detection = self.detection()

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


    class camera:
        def __init__(self, intrinsic_parameters=None, camera_index=None, image_width=None, image_height=None):
            self.intrinsic_parameters = intrinsic_parameters
            self.camera_index = camera_index
            self.image_width = image_width
            self.image_height = image_height

        def _calibrate(self, images, aruco_dict, parameters, criteria, size_of_marker):
            """
            This code will perform camera calibration using ArUco markers.
            It takes in a list of image paths containing ArUco markers and returns the camera matrix and distortion coefficients.
            arguments: images -- list of image paths containing ArUco markers
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
                if img is None:
                    print(f"  Warning: could not read image {fname}, skipping. Make sure the file exists and is an image.")
                    continue
                print(f"  Read image {fname}: {img.shape[1]}x{img.shape[0]} pixels")
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

        def calibrate(self, file_path, aruco_dict_type=cv2.aruco.DICT_4X4_50, size_of_marker=0.026, max_iterations=1000, termination_eps=1e-6, output_file=None):
            """
            - file_paths: List of paths to calibration images for each camera. Order matters and should correspond to camera numbers.
            - size_of_marker: The length of the aruco marker's side in meters.
            - output_file: The list names for the files where calibration results will be saved.
                - default: 'calibration_results_camX.npz' with X being the camera number.
            - aruco_dict_type: The type of aruco dictionary used for marker detection.
                - default: cv2.aruco.DICT_4X4_50 This is the aruco board available from the makerspace Aruco board.
            - max_iterations: The maximum number of iterations for corner refinement.
            - termination_eps: The desired accuracy for corner refinement. (this is between consecutive iterations shows convergences)
            """
            # Create aruco dictionary and parameters
            aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
            parameters = cv2.aruco.DetectorParameters()
            # Define criteria for corner refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, termination_eps)

            if output_file is None:
                output_file = [f'calibration_results_cam{i+1}.npz' for i in range(len(file_path))]

            results = []
            for i in range(len(output_file)):
                print(f"Calibrating camera {i+1}/{len(output_file)} using images from: {file_path[i]}")
                images = Cameras.image_paths_from_folder(file_path[i])
                camera_matrix, dist_coeffs = self._calibrate(images, aruco_dict, parameters, criteria, size_of_marker)
                # Save the calibration results
                np.savez(output_file[i], camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
                print(f"Saved calibration results to {output_file[i]}\n")
                results.append(np.load(output_file[i]))

            # Return the calibration results for all cameras
            return results
        
        def compute_depth_at_point(self, x, y):
            

    class detection:
        def __init__(self, object_detection_parameters):
            self.object_detection_parameters = object_detection_parameters

        def detect_objects_in_image(self, image):
            # Placeholder for object detection logic
            pass

        def get_bounding_boxes(self, detected_objects):
            # Placeholder for extracting bounding boxes from detected objects
            pass
