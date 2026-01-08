"""
This file creates a function for the stepper motors to move the robot forward, backward, left or right.
When moving forward or backward, both main motors move opposite directions 
to create movement in the same direction, since the rails are oriented oppositely.
Input tuple is in the form of (direction, distance in cm).
Possible directions are "front", "rear", "left", "right", "up", and "down".
"""
import time
import board
import numpy as np
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit

kit1 = MotorKit(i2c=board.I2C(address=0x60))
kit2 = MotorKit(i2c=board.I2C(address=0x61))
kit3 = MotorKit(i2c=board.I2C(address=0x62))

# Adjust motor based on position on robot, not direction it moves
motor_dict = {
    "rear_main": kit1.stepper1,
    "front_main": kit1.stepper2,
    "right_rail": kit2.stepper1,
    "left_rail": kit2.stepper2,
    "arm": kit3.stepper1,
}
# dictionary to map direction to motors and their movement direction
direction_dict = {
    "front": [("front_main", 1), ("rear_main", -1)],
    "rear": [("rear_main", 1), ("front_main", -1)],
    "left": [("left_rail", 1), ("right_rail", -1)],
    "right": [("right_rail", 1), ("left_rail", -1)],
    "up": [("arm", 1)],
    "down": [("arm", -1)],
}


diameter_wheel = 2.5 # in cm
circumference_wheel = diameter_wheel * np.pi
steps_per_revolution = 200  # Steps per full revolution of the stepper motor 200 is the standard for Nema 17
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

    for _step in range(max(abs(steps) for steps in steps_per_direction.values())):
        for motor_name, dir_multiplier in motors_to_move:
            motor = motor_dict[motor_name]
            step_direction = stepper.FORWARD if dir_multiplier > 0 else stepper.BACKWARD
            motor.onestep(direction=step_direction, style=stepper.SINGLE)
        time.sleep(speed)
    for motor_name, _ in motors_to_move:
        if motor_name != "arm":
            motor_dict[motor_name].release()
            


if __name__ == "__main__":
    move_cm(10, motor=kit1.stepper1)
    time.sleep(1)
    move_cm(-10, motor=kit1.stepper1)
    time.sleep(1)

    move_direction(speed=0.01, direction_to_move=[("front", 10), ("left", 5)])
    time.sleep(1)
    move_direction(speed=0.01, direction_to_move=[("rear", 10), ("right", 5)])
    time.sleep(1)

    

        


