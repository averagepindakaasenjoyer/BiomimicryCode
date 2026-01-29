import gpiozero as GPIO                                
from adafruit_motorkit import MotorKit
from gpiozero import OutputDevice
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
motor = OutputDevice(17, active_high=False, initial_value=False)



# Control the motor\
# motor.forward()  # stops motor weirdly enoug

while True:
    motor.on()      # stops motor









































    

