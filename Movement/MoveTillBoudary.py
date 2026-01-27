import threading
import time
import board
import numpy as np
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit
import busio
import board
from adafruit_motorkit import MotorKit
import gpiozero as GPIO                                
from adafruit_motorkit import MotorKit
from gpiozero import Motor
import busio
import board
from adafruit_motorkit import MotorKit
import cv2


i2c = busio.I2C(board.SCL, board.SDA)
kit1 = MotorKit(i2c=i2c, address=0x60)


# kit1 = MotorKit(i2c=board.I2C(), address=0x60)
kit2 = MotorKit(i2c=board.I2C(), address=0x61)
i2c = busio.I2C(board.SCL, board.SDA)
motor = Motor(forward=17, backward=27)


motor_dict = {
    "rails" : kit1.stepper1,
    "main": kit1.stepper2,
    "arm": kit2.stepper1,
}

direction_dict = {
    "front": [("main", 1), ("main", 1)],
    "rear": [("main", -1), ("main", -1)],
    "right": [("rails", 1), ("rails", 1)],
    "left": [("rails",-1), ("rails", -1)],
    "up": [("arm", -1)],
    "down": [("arm", 1)],
}

diameter_wheel = 2.5 # in cm
circumference_wheel = diameter_wheel * np.pi
steps_per_revolution = 200  # Steps per full revolution of the stepper motor 200 is the Nema 17 standard
steps_per_cm = steps_per_revolution / circumference_wheel
x_limit_cm= 47
y_limit_cm= 15.7
z_limit_cm=45.0


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

def move_cm(distance_cm, speed=0.03, motor=[kit1.stepper1], hold=True):
    """
    Move the robot forward or backward a certain distance in centimeters.
    
    :param distance_cm: Distance to move in centimeters. Positive for forward, negative for backward.
    :param speed: Delay between steps to control speed. Lower is faster.
    """
    steps = int(distance_cm * steps_per_cm)
    step_direction = stepper.FORWARD if steps > 0 else stepper.BACKWARD
    steps = abs(steps)

    for _ in range(steps):
        for i in range(len(motor)):
            motor[i].onestep(direction=step_direction, style=stepper.DOUBLE)
        time.sleep(speed)
    if not hold:
        for i in motor:
            i.release()

def release_all_motors():
    """
    Release all motors to stop holding their position.
    """
    for motor in motor_dict.values():
        motor.release()

def reset_robot_position():
    """
    Reset the robot's position by moving all motors to their initial positions.
    """
    
    thread1 = threading.Thread(target=move_cm, args=(
        -x_limit_cm, 0.02, [kit1.stepper2], False))
    thread2 = threading.Thread(target=move_cm, args=(
        -y_limit_cm, 0.02, [kit1.stepper1], False))
    thread3 = threading.Thread(target=move_cm, args=(
        -z_limit_cm, 0.02, [kit2.stepper1], False))
    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()

    release_all_motors()
    print("Robot position reset complete.")

def shake(times):
    """
    shake the vibration motor for a set amount of time
    """
    kit2.motor3.throttle = 1.0  # starts motor forward at full speed

    time.sleep(times)
    kit2.motor3.throttle = 0.0  # stops motor

def VanDeGraaf_move(times):
    """
    move the Van De Graaf motor for a set amount of time
    """
    motor.stop()  # stops motor

    time.sleep(times)
    motor.forward()  # stops motor

def DetectBoundary():
    """
    Use vision to detect red border and stop movement
    Border is possible at top, bottom, left, right else return none
    """
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red) 

    # detect red areas
    red_area = cv2.countNonZero(mask1)
    height, width, _ = frame.shape
    if red_area > 5000:  # threshold for detection
        # Check positions
        if np.any(mask1[0, :]):  # Top border
            cap.release()
            return "top"
        elif np.any(mask1[-1, :]):  # Bottom border
            cap.release()
            return "bottom"
        elif np.any(mask1[:, 0]):  # Left border
            cap.release()
            return "left"
        elif np.any(mask1[:, -1]):  # Right border
            cap.release()
            return "right"
    cap.release()
    return None
    





if __name__ == "__main__":
    # Example usage: Move forward 20 cm and right 10 cm simultaneously
    # ARM MIOVES UP 45 CM
    thread1 = threading.Thread(target=move_cm, args=(
        -40, 0, [kit2.stepper1], True))
    thread2 = threading.Thread(target=move_cm, args=(
        10, 0.02, [kit1.stepper1], False))
    thread3 = threading.Thread(target=move_cm, args=(
        10, 0.02, [kit1.stepper2], False))
    thread4 = threading.Thread(target=VanDeGraaf_move, args=(5,))
    thread5 = threading.Thread(target=shake, args=(5,))


    # thread1.start()
    
    # thread2.start()
    # thread3.start()
    # thread4.start()
    # thread5.start()

    
    # thread3.join()
    # thread2.join()
    # thread4.join()
    # thread5.join()
    # thread1.join()

    while not DetectBoundary():
        move_cm(1, 0.02, [kit1.stepper1, kit1.stepper2], False) 
