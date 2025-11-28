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
import torch


class Cameras:
    def __init__(self):
        self.depth_estimation = None
        self.detection = None 
        self.cameras = []
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

    def shoot_image(self, camera=None, picamera=False):
        """
        Capture an image from the specified camera.

        arguments:
            camera -- camera object to use (default is the first camera in the list)
            picamera -- boolean indicating if using Raspberry Pi Camera (default is False)
        returns: captured image
        """
        if camera is None:
            if len(self.cameras) == 0:
                raise ValueError("No cameras available to shoot image from.")
            camera = self.cameras[0]

        if picamera:
            from picamera import PiCamera
            from picamera.array import PiRGBArray
            camera = PiCamera()
            raw_capture = PiRGBArray(camera)
            camera.capture(raw_capture, format="bgr")
            image = raw_capture.array
            camera.close()
        else:
            cap = cv2.VideoCapture(camera.camera_index)
            ret, image = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError(f"Failed to capture image from camera index {camera.camera_index}")
        return image
    
    def shoot_images_from_all_cameras(self, picamera=False):
        """
        Capture images from all available cameras.

        arguments:
            picamera -- boolean indicating if using Raspberry Pi Camera (default is False)
        returns: list of captured images
        """
        images = []
        for cam in self.cameras:
            img = self.shoot_image(camera=cam, picamera=picamera)
            images.append(img)
        return images



    class camera:
        def __init__(self, intrinsic_parameters=None, camera_index=None, image_width=None, image_height=None, Camera_class=None):
            self.intrinsic_parameters = intrinsic_parameters
            self.camera_index = camera_index
            self.image_width = image_width
            self.image_height = image_height
            Camera_class.cameras.append(self)

        def get_camera_indices(self, max_index=10):
            """
            Detect available camera indices up to max_index.

            arguments:
                max_index -- maximum index to check (default is 10)
            returns: list of available camera indices
            """
            unavailable_indices = Cameras().camera_indices if hasattr(Cameras(), 'camera_indices') else []
            for index in range(max_index):
                cap = cv2.VideoCapture(index)
                if cap.isOpened() and index not in unavailable_indices:
                    available_index = index
                    cap.release()
            self.camera_indices = index
            Cameras().camera_indices = unavailable_indices + [available_index]
            return available_index

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
        
        def camera_info(self):
            """
            Returns information about the camera.
            returns: dictionary containing camera index, intrinsic parameters, image width and height
            """
            info = {
                'camera_index': self.camera_index,
                'intrinsic_parameters': self.intrinsic_parameters,
                'image_width': self.image_width,
                'image_height': self.image_height
            }
            return info
        
        def set_parameters(self, intrinsic_parameters=None, camera_index=None, image_width=None, image_height=None):
            """
            Set or update camera parameters.
            arguments: intrinsic_parameters -- new intrinsic parameters
                       camera_index -- new camera index
                       image_width -- new image width
                       image_height -- new image height
            """
            if intrinsic_parameters is not None:
                self.intrinsic_parameters = intrinsic_parameters
            if camera_index is not None:
                self.camera_index = camera_index
            if image_width is not None:
                self.image_width = image_width
            if image_height is not None:
                self.image_height = image_height

        
    class depth_estimation:
        def __init__(self, stereo_parameters=None):
            self.stereo_parameters = stereo_parameters

        def compute_depth_map(self, left_image, right_image, Cameras):
            """
            Compute depth map from stereo images.
            arguments: left_image -- left image of the stereo pair
                       right_image -- right image of the stereo pair
                       camera_parameters -- dictionary containing camera matrices and distortion coefficients
            returns: depth_map
            """
            print("Amount cameras: ",Cameras.cameras)
            if len(Cameras.cameras) < 2:
                raise ValueError("At least two cameras are required for depth estimation.")
        

            left_camera_matrix = Cameras.cameras[0].intrinsic_parameters
            right_camera_matrix = Cameras.cameras[1].intrinsic_parameters

            # Create StereoBM object
            stereo = cv2.StereoSGBM_create(numDisparities=16*5, blockSize=5)

            # Compute disparity map
            disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16.0

            # Convert disparity to depth
            focal_length = left_camera_matrix[0, 0]  # Assuming fx is at (0,0)
            baseline = np.linalg.norm(self.stereo_parameters['T'])

            with np.errstate(divide='ignore'):
                depth_map = (focal_length * baseline) / disparity_map
                depth_map[disparity_map == 0] = 0

            return depth_map

        def shoot_and_depth_calc(self, Cameras):
            """shoots images from the first two cameras and computes the depth map."""
            if len(Cameras.cameras) < 2:
                raise ValueError("At least two cameras are required for depth estimation.")

            left_image = Cameras().shoot_image(camera=Cameras.cameras[0])
            right_image = Cameras().shoot_image(camera=Cameras.cameras[1])

            return self.compute_depth_map(left_image, right_image, Cameras)
            
        def compute_depth_at_point(self, depth_map, x, y):
            """
            Compute the depth at a specific point in the depth map.
            arguments: depth_map -- computed depth map
                       x -- x coordinate of the point
                       y -- y coordinate of the point
            returns: depth at the specified point
            """
            if x < 0 or x >= depth_map.shape[1] or y < 0 or y >= depth_map.shape[0]:
                raise ValueError("Coordinates are out of bounds of the depth map.")
            return depth_map[y, x]

    class detection:
        def __init__(self, object_detection_parameters):
            self.object_detection_parameters = object_detection_parameters
            self.confidence_threshold = 0.5
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        def Yolo_flower_detection(self, image, yolo_model, visual_feedback=False):
            """
            function calls on yolo algorithm to detect flowers in the image.
            arguments: image -- input image for object detection, yolo_model -- loaded yolo model, visual_feedback -- flag to indicate if visual feedback is needed
            returns: tuple of detected bounding boxes and confidence scores and class_ids
            """
            results = yolo_model.predict(source=image, conf=self.confidence_threshold, device=self.device, verbose=False)

            r = results[0]

            if hasattr(r, "boxes") and r.boxes is not None:
                boxes_obj = r.boxes

                xyxy = getattr(boxes_obj, "xyxy", None) # x1, y1, x2, y2
                confs = getattr(boxes_obj, "conf", None)
                class_ids = getattr(boxes_obj, "cls", None)

                def to_np(x):
                    if x is None:
                        return None
                    try:
                        return x.cpu().numpy()
                    except Exception:
                        return np.array(x)

                xyxy = to_np(xyxy)
                confs = to_np(confs)
                class_ids = to_np(class_ids)

                if visual_feedback:
                    names = getattr(yolo_model, "names", {})

                    # Draw bounding boxes on the image
                    for i, box in enumerate(xyxy):
                        x1, y1, x2, y2 = map(int, box[:4])
                        score = float(confs[i]) if confs is not None else 0.0
                        cls_id = int(class_ids[i]) if class_ids is not None else -1
                        label = f"{names.get(cls_id, str(cls_id))} {score:.2f}"
                        color = (0, 255, 0)
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(image, label, (x1, max(y1 - 6, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    
                    # Print concise summary for debugging
                    n = 0 if xyxy is None else xyxy.shape[0]
                    print(f"Detections: {n} boxes")
                    if xyxy is not None:
                        print("xyxy:\n", xyxy)
                    if confs is not None:
                        print("confs:\n", confs)
                    if class_ids is not None:
                        print("class_ids:\n", class_ids)
                    print("class names mapping:\n", names)
                        
                    return image
                
                if xyxy is not None:
                    return (xyxy, confs, class_ids) # tuple: (bboxes, conf, class)
            else:
                print("No boxes found in results.")
                return ([], [], [])
            

        def center_bounding_boxes(self, bounding_boxes, camera):
            """
            function to get the center points of bounding boxes.
            arguments: bounding_boxes -- list of bounding boxes
                       camera -- camera object to get camera info to transform percentages to pixels
            returns: list of center points
            """
            centers = []
            camera_info = camera.camera_info()
            for box in bounding_boxes:
                x, y, w, h = box
                x = x * camera_info['image_width']
                y = y * camera_info['image_height']
                w = w * camera_info['image_width']
                h = h * camera_info['image_height']
                center_x = x + w / 2
                center_y = y + h / 2
                centers.append((center_x, center_y))
            return centers
        

        def get_movement_distances(self, flowers_centers):
            """
            Function to calculate the x y axis movement in pixels from center of frame to flower center.
            arguments: flowers_centers -- list of flower center points
            returns: list of tuples containing x and y distances
            """
            movements = []
            for center in flowers_centers:
                center_x, center_y = center
                movement_x = center_x - (self.object_detection_parameters['image_width'] // 2)
                movement_y = center_y - (self.object_detection_parameters['image_height'] // 2)
                movements.append((movement_x, movement_y))
            return movements
        def detect_circel(self, image):
            """
            Function to detect yellow circles in the image.
            arguments: image -- input image for circle detection
            returns: coordinates of detected circles in percentages of image size
            """
            recognized_circles = []
            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Define yellow color range in HSV
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            # Create a mask for yellow color
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            # Apply Hough Circle Transform
            # apply the mask to the original image and convert to grayscale for HoughCircles
            masked = cv2.bitwise_and(image, image, mask=mask)
            gray_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

            # reduce noise — larger kernel helps remove small spurious blobs
            gray_masked = cv2.medianBlur(gray_masked, 7)
            gray_masked = cv2.GaussianBlur(gray_masked, (9, 9), 0)

            # set min/max radius relative to image size to prefer one large circle
            h, w = image.shape[:2]
            min_r = max(40, int(min(h, w) * 0.1))
            max_r = int(min(h, w) * 0.5)

            # Use HoughCircles on the preprocessed grayscale image with tuned parameters:
            # - dp > 1 to reduce resolution and false positives
            # - minDist large so only one circle is found
            # - higher param1 for Canny edge detection, higher param2 to require stronger accumulator votes
            circles = cv2.HoughCircles(gray_masked, cv2.HOUGH_GRADIENT,
                                       dp=1.5,
                                       minDist=max(h, w) // 2,
                                       param1=100,
                                       param2=40,
                                       minRadius=min_r,
                                       maxRadius=max_r)
            print("Detected circles:", circles)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center_x = i[0] / image.shape[1]
                    center_y = i[1] / image.shape[0]
                    recognized_circles.append((center_x, center_y))

            X_shape, Y_shape = image.shape[1], image.shape[0]
            precentage_circles = [(x / X_shape, y / Y_shape) for (x, y) in recognized_circles]
            return precentage_circles





if __name__ == "__main__":
    # Load image and run circle detection without relying on camera objects
    image_path = r"C:\Users\Matth\School2025\Biomimicry\BiomimicryCode\GeleCirkel.JPG"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    detector = Cameras.detection(object_detection_parameters={'image_width': image.shape[1], 'image_height': image.shape[0]})
    circles = detector.detect_circel(image)
    camera_upper = Cameras()
    camera1 = camera_upper.camera(camera_index=0, intrinsic_parameters=np.eye(3), image_width=image.shape[1], image_height=image.shape[0],Camera_class=camera_upper)
    camera2 = camera_upper.camera(camera_index=1, intrinsic_parameters=np.eye(3), image_width=image.shape[1], image_height=image.shape[0],Camera_class=camera_upper)
    print("Detected circles (in percentages):", circles)
    movements = detector.get_movement_distances(circles)
    print("Movement distances (in pixels):", movements)
    print("Image width:", image.shape[1], "Image height:", image.shape[0])
    left_image = image
    right_image = image
    depth_estimator = Cameras.depth_estimation(stereo_parameters={'T': np.array([0.1, 0, 0])})
    depth_map = depth_estimator.compute_depth_map(left_image, right_image, camera_upper)
    print("Depth map shape:", depth_map.shape)
    depth_at_center = depth_estimator.compute_depth_at_point(depth_map, image.shape[1]//2, image.shape[0]//2, )
    print("Depth at image center:", depth_at_center)

