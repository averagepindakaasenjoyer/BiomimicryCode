"""
This code consist of the code needed for the movement of the robot.
    It uses stepper motors to move the robot forward, backward, and to turn.
"""
import time
import board
import numpy as np
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit



Diameter_Wheel = 2.5 # in cm
Circumference_Wheel = Diameter_Wheel * np.pi
Steps_Per_Revolution = 200  # Steps per full revolution of the stepper motor 200 is the Nema 17 standard
Steps_Per_Cm = Steps_Per_Revolution / Circumference_Wheel

def move_cm(distance_cm, speed=0.01, motor=kit.stepper1):
    """
    Move the robot forward or backward a certain distance in centimeters.
    
    :param distance_cm: Distance to move in centimeters. Positive for forward, negative for backward.
    :param speed: Delay between steps to control speed. Lower is faster.
    """
    steps = int(distance_cm * Steps_Per_Cm)
    step_direction = stepper.FORWARD if steps > 0 else stepper.BACKWARD
    steps = abs(steps)

    for _ in range(steps):
        motor.onestep(direction=step_direction, style=stepper.SINGLE)
        time.sleep(speed)
    motor.release()


if __name__ == "__main__":
    kit = MotorKit(i2c=board.I2C())
    # Example usage: Move down 10 cm
    move_cm(10, motor=kit.stepper1)
    time.sleep(1)
    # Move up 10 cm
    move_cm(-10, motor=kit.stepper1)
    time.sleep(1)

    

        


