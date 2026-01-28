"""
Pi Server for Stereo Vision Robot Control

Sends stereo camera frames to laptop for processing.
Receives motor commands from laptop and executes them.
"""

import socket
import cv2
import pickle
import struct
import time
import threading
import board
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit
from gpiozero import OutputDevice 

# Network configuration
PORT = 8000

# Camera indices (adjust based on your setup)
CAM_LEFT = 0
CAM_RIGHT = 2

# =============================
# Motor Hardware Setup
# =============================
print("[Pi] Initializing motor controllers...")
kit1 = MotorKit(i2c=board.I2C(), address=0x60)
kit2 = MotorKit(i2c=board.I2C(), address=0x61)

# GPIO motor for Van de Graaf (pins 17 and 27)
van_de_graaf_gpio_motor = OutputDevice(17, active_high=False, initial_value=False)

motor_dict = {
    "rails": kit1.stepper1,
    "main": kit1.stepper2,
    "arm": kit2.stepper1,
}

# Continuous motors (throttle-based, for timing-controlled operations)
continuous_motor_dict = {
    "vibrate": kit2.motor3,  # Vibration motor on kit2.motor3
}

# Van de Graaf uses separate GPIO motor
van_de_graaf_motor = van_de_graaf_gpio_motor

# Motor step delay
STEP_DELAY = 0.01
STEP_DELAY_ARM = 0.01

# Vibration throttle setting
VIBRATE_THROTTLE = 1.0  # Full speed for vibration motor

# =============================
# Motor Control Functions
# =============================
def release_all_motors():
    """Release all motors to prevent overheating."""
    for m in motor_dict.values():
        try:
            m.release()
        except Exception:
            pass

def motor_step_worker(motor_obj, steps, step_delay=STEP_DELAY, style=stepper.DOUBLE, hold=False):
    """Drive motor for given steps (signed).
    
    Args:
        motor_obj: Motor object to control
        steps: Number of steps (positive/negative for direction)
        step_delay: Delay between steps
        style: Stepping style (SINGLE, DOUBLE, etc.)
        hold: If True, don't release motor after stepping
    """
    if steps == 0:
        return
    direction = stepper.FORWARD if steps > 0 else stepper.BACKWARD
    steps = abs(int(steps))
    try:
        for _ in range(steps):
            motor_obj.onestep(direction=direction, style=style)
            time.sleep(step_delay)
    except Exception as e:
        print(f"[Pi] Motor error: {e}")
    finally:
        # Only release if not in hold mode
        if not hold:
            try:
                motor_obj.release()
            except Exception:
                pass

def continuous_motor_worker(motor_obj, motor_name, duration_ms, throttle=1.0):
    """Run continuous motor for specified duration in milliseconds.
    
    Args:
        motor_obj: Motor object to control (continuous motor)
        motor_name: Name of motor (for logging)
        duration_ms: Duration to run motor in milliseconds
        throttle: Throttle value (0.0 to 1.0)
    """
    duration_s = duration_ms / 1000.0
    try:
        motor_obj.throttle = throttle
        time.sleep(duration_s)
    except Exception as e:
        print(f"[Pi] Continuous motor error ({motor_name}): {e}")
    finally:
        try:
            motor_obj.throttle = 0.0  # Stop motor
        except Exception:
            pass

def van_de_graaf_worker(duration_ms):
    """Run Van de Graaf GPIO motor for specified duration in milliseconds.
    
    Args:
        duration_ms: Duration to run motor in milliseconds
    """
    duration_s = duration_ms / 1000.0
    print(f"[Pi] Van de Graaf starting for {duration_ms}ms ({duration_s}s)")
    
    try:
        van_de_graaf_motor.off()  # Ensure stopped first
        time.sleep(0.1)  # Brief delay
        print(f"[Pi] Van de Graaf motor.on() starting...")
        van_de_graaf_motor.on()  # Start motor
        time.sleep(duration_s)
        print(f"[Pi] Van de Graaf motor.off() called after {duration_s}s")
    except Exception as e:
        print(f"[Pi] Van de Graaf motor error: {e}")
    finally:
        # Ensure motor stops - don't silence exceptions
        van_de_graaf_motor.off()
        print(f"[Pi] Van de Graaf motor stopped in finally block")

def execute_motor_command(command_dict):
    """
    Execute motor movements based on command dictionary.
    
    Args:
        command_dict: Dictionary with motor names as keys and values
                     Stepper motors: step counts (positive/negative for direction)
                     Continuous motors (vibrate, van_de_graaf): duration in milliseconds
                     Supports special keys:
                       - _action: 'release_all' to release all motors
                       - _hold_motors: list of stepper motor names to keep held after movement
                     e.g., {'rails': 100, 'main': -50, 'arm': 20, '_hold_motors': ['arm']}
                     e.g., {'vibrate': 500, 'van_de_graaf': 200}

    """
    if not command_dict or len(command_dict) == 0:
        print("[Pi] Empty command dict, no motors to execute")
        return
    
    print(f"[Pi] EXECUTING MOTOR COMMAND: {command_dict}")
    
    # Handle special action commands
    action = command_dict.get("_action")
    if action == "release_all":
        print("[Pi] Releasing all motors")
        release_all_motors()
        return
    
    # Extract hold_motors list (motors that should NOT be released after movement)
    hold_motors = set(command_dict.get("_hold_motors", []))
    
    # Filter out special keys
    motor_commands = {k: v for k, v in command_dict.items() if not k.startswith("_")}
    
    print(f"[Pi] Motor commands after filtering: {motor_commands}")
    
    if not motor_commands:
        return
    
    print(f"[Pi] Executing motor command: {motor_commands}")
    if hold_motors:
        print(f"[Pi] Motors to hold: {hold_motors}")
    
    threads = []
    for motor_name, value in motor_commands.items():
        # Check if this is a stepper motor
        motor_obj = motor_dict.get(motor_name)
        if motor_obj is not None:
            # Stepper motor
            steps = value
            if steps == 0:
                continue
            
            # Use appropriate delay for arm motor
            delay = STEP_DELAY_ARM if motor_name == "arm" else STEP_DELAY
            
            # Check if this motor should be held
            hold = motor_name in hold_motors
            
            t = threading.Thread(target=motor_step_worker, args=(motor_obj, steps, delay, stepper.DOUBLE, hold))
            t.daemon = True
            threads.append(t)
            t.start()
        elif motor_name == "van_de_graaf":
            # Van de Graaf GPIO motor (special case)
            duration_ms = value
            if duration_ms <= 0:
                continue
            
            t = threading.Thread(target=van_de_graaf_worker, args=(duration_ms,))
            t.daemon = True
            threads.append(t)
            t.start()
        else:
            # Check if this is a continuous motor (throttle-based)
            cont_motor_obj = continuous_motor_dict.get(motor_name)
            if cont_motor_obj is not None:
                # Continuous motor (value is duration in milliseconds)
                duration_ms = value
                if duration_ms <= 0:
                    continue
                
                # Get appropriate throttle
                throttle = VIBRATE_THROTTLE
                
                t = threading.Thread(target=continuous_motor_worker, args=(cont_motor_obj, motor_name, duration_ms, throttle))
                t.daemon = True
                threads.append(t)
                t.start()
            else:
                print(f"[Pi] Warning: Unknown motor '{motor_name}'")
    
    # Wait for all motors to complete
    for t in threads:
        t.join()
    
    print("[Pi] Motor command complete")

# =============================
# Network Communication
# =============================
def send_stereo_frames(conn, frame_left, frame_right):
    """Send stereo frame pair to laptop."""
    # Compress both frames to JPEG
    _, buffer_left = cv2.imencode('.jpg', frame_left, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    _, buffer_right = cv2.imencode('.jpg', frame_right, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    
    # Package both frames together
    data = pickle.dumps({
        'left': buffer_left,
        'right': buffer_right,
        'timestamp': time.time()
    })
    
    # Send with size header
    message = struct.pack("Q", len(data)) + data
    conn.sendall(message)

def receive_motor_command(conn):
    """Receive motor command from laptop."""
    try:
        conn.setblocking(False)  # Non-blocking mode
        try:
            data = conn.recv(4096)
        except BlockingIOError:
            # No data available
            return None
        finally:
            conn.setblocking(True)  # Back to blocking
        
        if not data:
            return None
        
        command_dict = pickle.loads(data)
        print(f"[Pi] RECEIVED MOTOR COMMAND: {command_dict}")
        return command_dict
    except Exception as e:
        print(f"[Pi] Error receiving command: {e}")
        return None

# =============================
# Main Server Loop
# =============================
def main():
    # Set up socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', PORT))
    server_socket.listen(1)
    
    try:
        while True:  # Loop to handle multiple connections
            print(f"[Pi] Waiting for connection on port {PORT}...")
            
            conn, addr = server_socket.accept()
            print(f"[Pi] Connected to {addr}")
            
            # Open stereo cameras
            print(f"[Pi] Opening cameras {CAM_LEFT} and {CAM_RIGHT}...")
            cap_left = cv2.VideoCapture(CAM_LEFT)
            cap_right = cv2.VideoCapture(CAM_RIGHT)
            
            if not cap_left.isOpened() or not cap_right.isOpened():
                print("[Pi] ERROR: Could not open cameras!")
                conn.close()
                continue  # Wait for next connection
            
            # Set camera properties
            for cap in [cap_left, cap_right]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("[Pi] Cameras initialized successfully")
            
            try:
                frame_count = 0
                while True:
                    # Capture stereo frames
                    ret_l, frame_left = cap_left.read()
                    ret_r, frame_right = cap_right.read()
                    
                    if not ret_l or not ret_r:
                        print("[Pi] Failed to read frames")
                        time.sleep(0.1)
                        continue
                    
                    # Send frames to laptop
                    try:
                        send_stereo_frames(conn, frame_left, frame_right)
                        frame_count += 1
                        if frame_count % 30 == 0:
                            print(f"[Pi] Sent {frame_count} frame pairs")
                    except Exception as e:
                        print(f"[Pi] Error sending frames: {e}")
                        print("[Pi] Connection broken, cleaning up...")
                        break
                    
                    # Receive and execute motor commands
                    command = receive_motor_command(conn)
                    if command:
                        execute_motor_command(command)
                    
                    time.sleep(0.01)  # Small delay to prevent overwhelming the network
            
            except Exception as e:
                print(f"[Pi] Error during connection: {e}")
            finally:
                # Cleanup for this connection
                cap_left.release()
                cap_right.release()
                conn.close()
                release_all_motors()
                print("[Pi] Connection cleanup complete")
    
    except KeyboardInterrupt:
        print("[Pi] Interrupted by user")
    finally:
        server_socket.close()
        print("[Pi] Server shutdown complete")

if __name__ == "__main__":
    main()
