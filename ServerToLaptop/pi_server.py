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

# Network configuration
PORT = 8000

# Camera indices (adjust based on your setup)
CAM_LEFT = 1
CAM_RIGHT = 2

# =============================
# Motor Hardware Setup
# =============================
print("[Pi] Initializing motor controllers...")
kit1 = MotorKit(i2c=board.I2C(), address=0x60)
kit2 = MotorKit(i2c=board.I2C(), address=0x61)

motor_dict = {
    "rails": kit1.stepper1,
    "main": kit1.stepper2,
    "arm": kit2.stepper1,
}

# Motor step delay
STEP_DELAY = 0.01
STEP_DELAY_ARM = 0.01

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

def motor_step_worker(motor_obj, steps, step_delay=STEP_DELAY, style=stepper.DOUBLE):
    """Drive motor for given steps (signed)."""
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
        try:
            motor_obj.release()
        except Exception:
            pass

def execute_motor_command(command_dict):
    """
    Execute motor movements based on command dictionary.
    
    Args:
        command_dict: Dictionary with motor names as keys and step counts as values
                     e.g., {'rails': 100, 'main': -50, 'arm': 20}
    """
    if not command_dict or len(command_dict) == 0:
        return
    
    print(f"[Pi] Executing motor command: {command_dict}")
    
    threads = []
    for motor_name, steps in command_dict.items():
        motor_obj = motor_dict.get(motor_name)
        if motor_obj is None:
            print(f"[Pi] Warning: Unknown motor '{motor_name}'")
            continue
        
        if steps == 0:
            continue
        
        # Use appropriate delay for arm motor
        delay = STEP_DELAY_ARM if motor_name == "arm" else STEP_DELAY
        
        t = threading.Thread(target=motor_step_worker, args=(motor_obj, steps, delay))
        t.daemon = True
        threads.append(t)
        t.start()
    
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
        data = conn.recv(4096)
        if not data:
            return None
        
        command_dict = pickle.loads(data)
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
        server_socket.close()
        return
    
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
                break
            
            # Receive and execute motor commands
            command = receive_motor_command(conn)
            if command:
                execute_motor_command(command)
            
            time.sleep(0.01)  # Small delay to prevent overwhelming the network
    
    except KeyboardInterrupt:
        print("[Pi] Interrupted by user")
    except Exception as e:
        print(f"[Pi] Error: {e}")
    finally:
        cap_left.release()
        cap_right.release()
        conn.close()
        server_socket.close()
        release_all_motors()
        print("[Pi] Cleanup complete")

if __name__ == "__main__":
    main()
