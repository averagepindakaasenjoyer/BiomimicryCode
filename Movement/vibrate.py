import gpiozero as GPIO                                
from adafruit_motorkit import MotorKit
from gpiozero import Motor
import busio
import board
from adafruit_motorkit import MotorKit
import time

# to use Raspberry Pi board pin numbers
# Create a motor on GPIO pins 17 and 27
i2c = busio.I2C(board.SCL, board.SDA)
kit1 = MotorKit(i2c=i2c, address=0x60)


# kit1 = MotorKit(i2c=board.I2C(), address=0x60)
kit2 = MotorKit(i2c=board.I2C(), address=0x61)

def shake(timimg):
    """
    shake the vibration motor for a set amount of time
    """
    kit2.motor3.throttle = 1.0  # starts motor forward at full speed

    time.sleep(timimg)

    kit2.motor3.throttle = 0.0  # stops motor

if __name__ == "__main__":
    # Example usage: Shake for 2 seconds
    shake(5)

    