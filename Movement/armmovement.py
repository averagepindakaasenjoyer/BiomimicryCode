"""
This code consist of the code needed for the movement of the robot.
    It uses stepper motors to move the robot forward, backward, and to turn.
"""
import time
import board
import numpy as np
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit


kit1 = MotorKit(i2c=board.I2C(), address=0x60)
kit2 = MotorKit(i2c=board.I2C(), address=0x61)


Diameter_Wheel = 2.5 # in cm
Circumference_Wheel = Diameter_Wheel * np.pi
Steps_Per_Revolution = 200  # Steps per full revolution of the stepper motor 200 is the Nema 17 standard
Steps_Per_Cm = Steps_Per_Revolution / Circumference_Wheel

def move_cm(distance_cm, speed=0.03, motor=[kit1.stepper1], hold=True):
    """
    Move the robot forward or backward a certain distance in centimeters.
    
    :param distance_cm: Distance to move in centimeters. Positive for forward, negative for backward.
    :param speed: Delay between steps to control speed. Lower is faster.
    """
    steps = int(distance_cm * Steps_Per_Cm)
    step_direction = stepper.FORWARD if steps > 0 else stepper.BACKWARD
    steps = abs(steps)

    for _ in range(steps):
        for i in range(len(motor)):
            motor[i].onestep(direction=step_direction, style=stepper.DOUBLE)
        time.sleep(speed)
    if not hold:
        for i in motor:
            i.release()


if __name__ == "__main__":
    # Example usage: Move down 10 cm
    move_cm(-40, motor=[kit1.stepper2, kit1.stepper1], hold=False)
    time.sleep(2)
    # Move up 10 cm
    move_cm(400, motor=[kit1.stepper2, kit1.stepper1], hold=False)
    time.sleep(1)

    

        


