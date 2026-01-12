"""
This code consist of the code needed for the movement of the robot based on an image.
    It uses stepper motors to move the robot forward, backward, and to turn.
    It uses an USB camera to capture images and process them to determine movement.
"""
"""
This code consist of the code needed for the movement of the robot.
    It uses stepper motors to move the robot forward, backward, and to turn.
"""
import time
import board
import numpy as np
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit
import cv2
import threading



kit1 = MotorKit(i2c=board.I2C(), address=0x60)
kit2 = MotorKit(i2c=board.I2C(), address=0x61)
kit3 = MotorKit(i2c=board.I2C(), address=0x62)
motor_dict = {
    "rear_main": kit1.stepper1,
    "front_main": kit1.stepper2,
    "right_rail": kit2.stepper1,
    "left_rail": kit2.stepper2,
    "arm": kit3.stepper1,
}

direction_dict = {
    "front": [("rear_main", -1), ("front_main", 1)],
    "rear": [("rear_main", 1), ("front_main", -1)],
    "left": [("left_rail", 1), ("right_rail", -1)],
    "right": [("right_rail", 1), ("left_rail", -1)],
    "up": [("arm", 1)],
    "down": [("arm", -1)],
}


diameter_wheel = 2.5 # in cm
circumference_wheel = diameter_wheel * np.pi
steps_per_revolution = 200  # Steps per full revolution of the stepper motor 200 is the Nema 17 standard
steps_per_cm = steps_per_revolution / circumference_wheel

def move_cm(distance_cm, speed=0.01, motor=[kit1.stepper1]):
    """
    Move the robot forward or backward a certain distance in centimeters.
    
    :param distance_cm: Distance to move in centimeters. Positive for forward, negative for backward.
    :param speed: Delay between steps to control speed. Lower is faster.
    """
    steps = int(distance_cm * steps_per_cm)
    step_direction = stepper.FORWARD if steps > 0 else stepper.BACKWARD
    steps = abs(steps)

    for _ in range(steps):
        motor.onestep(direction=step_direction, style=stepper.SINGLE)
        time.sleep(speed)
    motor.release()

def move_direction(speed=0.01, direction_to_move=[("front", 10)]):
    """
    Move a list of motors forward or backward a certain distance in centimeters.
    
    :param speed: Delay between steps to control speed. Lower is faster.
    :param direction_to_move: The directions and distances to move.
    """
    motors_to_move = []
    steps_per_direction = {}

    for direction, distance_cm in direction_to_move:
        motors_to_move.extend([motor_name for motor_name in direction_dict.get(direction, [])])
        steps_per_direction[direction] = (distance_cm * steps_per_cm)

    for _step in range(int(max(abs(steps) for steps in steps_per_direction.values()))):
        for motor_name, dir_multiplier in motors_to_move:
            motor = motor_dict[motor_name]
            step_direction = stepper.FORWARD if dir_multiplier > 0 else stepper.BACKWARD
            motor.onestep(direction=step_direction, style=stepper.DOUBLE)
        time.sleep(speed)
    for motor_name, _ in motors_to_move:
        if motor_name != "arm":
            motor_dict[motor_name].release()
        

def detect_circle(image):
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
            # show the detected circles on the image for debugging
            for (x_perc, y_perc) in precentage_circles:
                x = int(x_perc * X_shape)
                y = int(y_perc * Y_shape)
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # center
                cv2.circle(image, (x, y), 10, (255, 0, 0), 2)  # radius
            return precentage_circles


def process_image():
    """
    Process the captured image to determine movement and determine motor actions.
    
    :param image: Captured image from the USB camera.
    :param motor: Stepper motor to control.
    retrun tuple (move_distanceX, move_distanceY)
    """
    # take picture from usb camera  
    cap = cv2.VideoCapture(0)    
    ret, image = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to capture image")
        return (0, 0)
    circles = detect_circle(image)
    image_height, image_width = image.shape[:2]
    if circles:
        # only first circle is considered
        circle_x, circle_y = circles[0]
        # Convert percentages back to pixel coordinates migth be useful for debugging
        circle_x_pixel = int(circle_x * image_width)
        circle_y_pixel = int(circle_y * image_height)
        # Determine movement based on circle position
        center_x = image_width // 2
        center_y = image_height // 2
        move_distanceX = circle_x_pixel - center_x
        move_distanceY = circle_y_pixel - center_y
        return move_distanceX, move_distanceY
    return (0, 0)
        
def get_moving_direction(DistanceX, DistanceY):
    """
    This function will return the motion for the mai movement and the rails, 
    based on the distance of x and y it in pixels receives. It will move further if the number is higher
    and move less distance for a lower number. This is based on the formula:
    X/10, y/10 ???
    
    :param DistanceX: This is the distance needed to move in the X axis, this axis is controled by main movement
    :param DistanceY: This is the distance needed to move in the Y axis, this axis is controled by rail movement

    return:
        dict: Direction dict, but with correct values

    """
    moving_dict = direction_dict.copy()
    # main movement
    if DistanceX > 20:
        moving_dict["front"] = [("rear_main", -abs(DistanceX)//10), ("front_main", abs(DistanceX)//10)]
    elif DistanceX < -20:
        moving_dict["rear"] = [("rear_main", abs(DistanceX)//10), ("front_main", -abs(DistanceX)//10)]
    
    # rail movement
    if DistanceY > 20:
        moving_dict["left"] = [("left_rail", abs(DistanceY)//10), ("right_rail", -abs(DistanceY)//10)]
    elif DistanceY < -20:
        moving_dict["right"] = [("right_rail", abs(DistanceY)//10), ("left_rail", -abs(DistanceY)//10)]


    return  moving_dict




if __name__ == "__main__":  
    while True:
        DistanceX, DistanceY = process_image()  
        print(f"DistanceX: {DistanceX}, DistanceY: {DistanceY}")
        # good till here, distance x and y are distnaces in pixels from center of image, assume camra is centered on robot
        moving_dict = get_moving_direction(DistanceX, DistanceY)
        # move in threads for each motor in main movement and rail movement
        threads = []
        if "front" in moving_dict:
            t1 = threading.Thread(target=move_direction, args=(0.01, [("front", moving_dict["front"][0][1])]))
            threads.append(t1)
            t1.start()
        if "rear" in moving_dict:
            t2 = threading.Thread(target=move_direction, args=(0.01, [("rear", moving_dict["rear"][0][1])]))
            threads.append(t2)
            t2.start()
        if "left" in moving_dict:
            t3 = threading.Thread(target=move_direction, args=(0.01, [("left", moving_dict["left"][0][1])]))
            threads.append(t3)
            t3.start()
        if "right" in moving_dict:
            t4 = threading.Thread(target=move_direction, args=(0.01, [("right", moving_dict["right"][0][1])]))
            threads.append(t4)
            t4.start()

        # wait for all threads to finish
        for t in threads:
            t.join()

        