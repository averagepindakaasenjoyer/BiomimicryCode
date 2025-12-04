# Below imports all neccessary packages to make this Python Script run
import time
import board
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit

# Below initialises the variable kit to be our I2C Connected Adafruit Motor HAT
kit = MotorKit(i2c=board.I2C())

# If you uncomment below it will start by de-energising the Stepper Motor,
# Worth noting the final state the stepper motor is in is what will continue.
# Energised Stepper Motors get HOT over time along with the electronic silicon drivers

# The below loop will run 500 times. Each loop it will move one step, clockwise, then pause for 0.01 seconds
# This will almost look like a smooth rotation.

for i in range(500):
    
    kit.stepper2.onestep()
    kit.stepper1.onestep()
    time.sleep(0.01)

# The below loop will run 500 times. Each loop it will move two step, anti-Clockwise, then pause for 0.01 seconds
# This will almost look like a smooth rotation.

for i in range(500):
    
    kit.stepper2.onestep(direction=stepper.BACKWARD, style=stepper.DOUBLE)
    kit.stepper1.onestep(direction=stepper.BACKWARD, style=stepper.DOUBLE)
    time.sleep(0.01)  

# The below loop will run 500 times. Each loop it will move a Micro-step, anti-Clockwise, then pause for 0.01 seconds
# This will rotate very very slowly.

for i in range(500):

    kit.stepper2.onestep(direction=stepper.BACKWARD, style=stepper.MICROSTEP)
    kit.stepper1.onestep(direction=stepper.BACKWARD, style=stepper.MICROSTEP)
    time.sleep(0.01)

# The below line will de-energise the Stepper Motor so it can freely move
kit.stepper2.release()

kit.stepper1.release()

