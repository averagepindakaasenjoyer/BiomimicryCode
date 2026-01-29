"""
Laptop Client - Interactive Manual Control & Demo Mode

Manual control mode:
- Type commands in terminal to control motors
- Commands: reset, move <motor> <steps>, arm <steps>, demo, quit

Demo mode:
- Automatic flower detection and tracking
- Robot explores and approaches flowers
"""

import socket
import cv2
import pickle
import struct
import numpy as np
import torch
from ultralytics import YOLO
import os
import time
import threading
from datetime import datetime
from pynput import keyboard

# Import helper functions
from helpers import (
    clamp, 
    get_current_position, 
    update_position, 
    reset_position, 
    print_position,
    clamp_movement_to_limits, 
    is_flower_already_visited, 
    mark_flower_as_visited,
    rotate_frame, 
    scale_intrinsics, 
    setup_stereo_rectification, 
    compute_stereo_disparity,
    compute_depth_map, 
    detect_flowers, 
    estimate_roi_depth, 
    select_target_flower,
    convert_offsets_to_motor_steps, 
    convert_depth_to_arm_steps, 
    print_help
)

# =============================
# Network Configuration
# =============================
PI_IP = "100.98.87.47"  # Replace with your Pi's IP
PORT = 8000

# Message type identifiers for framed protocol
MSG_TYPE_FRAME = 1      # Stereo frame data
MSG_TYPE_COMMAND = 2    # Motor command

# =============================
# Vision Configuration
# =============================
YOLO_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../current_best_yolo.pt"))
CALIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Cam_Params/stereo_charuco_calibration_16cm.npz'))

CONFIDENCE_THRESHOLD = 0.6
YOLO_CONF = 0.5
SCALE_FOR_MATCHING = 0.5

MIN_DEPTH = 0.25
MAX_DEPTH = 2.0
EXPECTED_DISTANCE = 0.40

SHOW_DEBUG = True
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Camera orientation settings
SWAP_CAMERAS = True  # Set to True if cameras are mounted upside-down/rotated
ROTATE_LEFT = 0 if SWAP_CAMERAS else 180    # Rotation for left camera in degrees
ROTATE_RIGHT = 180 if SWAP_CAMERAS else 0   # Rotation for right camera in degrees

# =============================
# Motor Parameters
# =============================
WHEEL_DIAMETER_CM = 2.5
CIRCUMFERENCE_CM = WHEEL_DIAMETER_CM * np.pi
STEPS_PER_REV = 200
STEPS_PER_CM = STEPS_PER_REV / CIRCUMFERENCE_CM
scale_move = 1.0  # Changed from 0.1 - was causing 0-step commands

# Calibration factors for each axis (adjust based on actual measurements)
RAILS_CALIBRATION = 0.67  # Rails was at 45cm when tracking showed 30cm (30/45 = 0.67)
MAIN_CALIBRATION = 1.0    # Main was accurate

PIXELS_PER_CM = 25.84
MAX_CM_PER_CYCLE = 10.0
MAX_STEPS_PER_CYCLE = int(MAX_CM_PER_CYCLE * STEPS_PER_CM)

WHEEL_DIAMETER_CM_ARM = 3.5
CIRCUMFERENCE_CM_ARM = WHEEL_DIAMETER_CM_ARM * np.pi
STEPS_PER_REV_ARM = 200
STEPS_PER_CM_ARM = STEPS_PER_REV_ARM / CIRCUMFERENCE_CM_ARM
ARM_CALIBRATION_FACTOR = 1.0  # Adjust if arm movement doesn't match commands
# If arm moves 7 cm when commanded 10 cm, set this to 10/7 ≈ 1.429
STEPS_PER_CM_ARM = STEPS_PER_CM_ARM * ARM_CALIBRATION_FACTOR

# Arm bottom position offset compensation
ARM_BOTTOM_OFFSET_CM = 1.5  # Extra distance needed when arm is at bottom
arm_is_at_bottom = True  # Track if arm is at bottom position

direction_dict = {
    "front": [("main", -1), ("main", -1)],
    "rear": [("main", 1), ("main", 1)],
    "right": [("rails", -1), ("rails", -1)],
    "left": [("rails", 1), ("rails", 1)],
    "up": [("arm", 1)],
    "down": [("arm", -1)],
}


# Debug mode
DEBUG_MOVEMENT = True  # Set to True to stop at flower and display movement calculations
DEBUG_SKIP_DEPTH = False  # Set to True to skip depth estimation and detection

# =============================
# Visited Flowers Tracking
# =============================
visited_flowers = []  # list of dicts: {"x": ..., "y": ..., "timestamp": ...}
FLOWER_VISIT_RADIUS_CM = 5.0  # Minimum distance (cm) to consider a flower as "new"

# =============================
# Load Models and Calibration
# =============================
print(f"[Laptop] Loading YOLO model from {YOLO_MODEL_PATH}...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[Laptop] Using device: {device}")
model = YOLO(YOLO_MODEL_PATH)

print(f"[Laptop] Loading calibration from {CALIB_PATH}...")
calib = np.load(CALIB_PATH)
K_L = calib["K_left"]
D_L = calib["D_left"]
K_R = calib["K_right"]
D_R = calib["D_right"]
R = calib["R"]
T = calib["T"]
baseline = np.linalg.norm(T)
print(f"[Laptop] Baseline: {baseline:.4f} m")

# =============================
# Global State
# =============================
client_socket = None
demo_mode = False
demo_stop_flag = False
current_frame_left = None
current_frame_right = None
frame_lock = threading.Lock()
stereo_initialized = False
mapLx, mapLy, mapRx, mapRy, Q, K_L_use = None, None, None, None, None, None
frame_thread_failed = False
shutdown_flag = False

# =============================
# Position Tracking
# =============================
# Coordinate system: (0,0) is rear-right (origin)
# x: rails position (cm) - 0 at right, increases going left, max 45 cm
# y: main position (cm) - 0 at rear, increases going forward, max 18 cm
# z: arm position (cm) - 0 at bottom/released, increases going up, max 20 cm

current_position = {"x": 0.0, "y": 0.0, "z": 0.0}  # Position in cm
position_lock = threading.Lock()

# Boundary limits in cm
LIMIT_X_MIN = 0.0       # Rightmost position
LIMIT_X_MAX = 45.0      # Leftmost position (rails: 45 cm)
LIMIT_Y_MIN = 0.0       # Rearmost position
LIMIT_Y_MAX = 18.0      # Frontmost position (main: 18 cm)
LIMIT_Z_MIN = 0.0       # Bottom/released position
LIMIT_Z_MAX = 20.0      # Highest position (arm: 20 cm)
# Arm tip offset from camera center (measured in pixels, converted to cm)
# Arm tip is at -130px left, +150px down from camera center
OFFSET_X_CM = 5.03      # Compensate for arm tip being left of center (move right)
OFFSET_Y_CM = 5.81     # Compensate for arm tip being below center (move backward)

# =============================
# Display Thread
# =============================
def display_thread():
    """Show side-by-side stereo frames during manual mode."""
    global shutdown_flag
    window_name = "MANUAL MODE - Stereo Cameras"
    try:
        while not shutdown_flag:
            with frame_lock:
                if current_frame_left is None or current_frame_right is None:
                    pass_frame = None
                else:
                    pass_frame = (
                        current_frame_left.copy(),
                        current_frame_right.copy()
                    )
            if pass_frame is None:
                time.sleep(0.05)
                continue
            frame_left, frame_right = pass_frame

            display_frame_left = frame_left.copy()
            display_frame_right = frame_right.copy()
            
            # Apply rotation if configured
            display_frame_left = rotate_frame(display_frame_left, ROTATE_LEFT)
            display_frame_right = rotate_frame(display_frame_right, ROTATE_RIGHT)

            cv2.putText(display_frame_left, "Left Camera", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame_right, "Right Camera", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            h_display = DISPLAY_HEIGHT // 2
            w_display = DISPLAY_WIDTH // 2
            display_frame_left = cv2.resize(display_frame_left, (w_display, h_display))
            display_frame_right = cv2.resize(display_frame_right, (w_display, h_display))

            display_combined = np.hstack([display_frame_right, display_frame_left])
            cv2.imshow(window_name, display_combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                shutdown_flag = True
                break
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass



# =============================
# Network Communication
# =============================
def receive_stereo_frames(client_socket, data, payload_size):
    """Receive stereo frame pair from Pi."""
    # Read header: 8 bytes (size) + 1 byte (type)
    header_size = payload_size + 1
    while len(data) < header_size:
        packet = client_socket.recv(4096)
        if not packet:
            return None, None, data
        data += packet
    
    packed_msg_size = data[:payload_size]
    msg_type = data[payload_size]
    data = data[header_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]
    
    # Verify this is a frame message
    if msg_type != MSG_TYPE_FRAME:
        print(f"[Laptop] WARNING: Expected frame (type {MSG_TYPE_FRAME}), got type {msg_type}")
        # Skip this message
        while len(data) < msg_size:
            packet = client_socket.recv(4096)
            if not packet:
                return None, None, data
            data += packet
        data = data[msg_size:]
        return None, None, data
    
    while len(data) < msg_size:
        packet = client_socket.recv(4096)
        if not packet:
            return None, None, data
        data += packet
    
    frame_data = data[:msg_size]
    data = data[msg_size:]
    
    frames_dict = pickle.loads(frame_data)
    
    frame_left = cv2.imdecode(frames_dict['left'], cv2.IMREAD_COLOR)
    frame_right = cv2.imdecode(frames_dict['right'], cv2.IMREAD_COLOR)
    
    return frame_left, frame_right, data

def send_motor_command(client_socket, command_dict):
    """Send motor command to Pi with typed framing protocol."""
    try:
        data = pickle.dumps(command_dict)
        # Pack: 8 bytes (size) + 1 byte (type) + payload
        message = struct.pack("QB", len(data), MSG_TYPE_COMMAND) + data
        print(f"[Laptop] SENDING MOTOR COMMAND: {command_dict} ({len(data)} bytes, type={MSG_TYPE_COMMAND})")
        client_socket.sendall(message)
        print(f"[Laptop] Motor command sent successfully")
    except Exception as e:
        print(f"[Laptop] Error sending command: {e}")

# =============================
# Pollination Motor Control
# =============================
def vibrate_motor(client_socket, duration_ms):
    """Control vibration motor for specified duration in milliseconds.
    
    Args:
        client_socket: Connected socket to Pi
        duration_ms: Duration to run vibration motor in milliseconds
    """
    motor_command = {"vibrate": duration_ms}
    print(f"[POLLINATE] Vibrating for {duration_ms}ms")
    send_motor_command(client_socket, motor_command)

def van_de_graaf_motor(client_socket, duration_ms):
    """Control van de Graaf generator motor for specified duration in milliseconds.
    
    Args:
        client_socket: Connected socket to Pi
        duration_ms: Duration to run van de Graaf motor in milliseconds
    """
    motor_command = {"van_de_graaf": duration_ms}
    print(f"[POLLINATE] Running van de Graaf for {duration_ms}ms")
    send_motor_command(client_socket, motor_command)

def pollinate(client_socket, vibrate_duration_ms=500, van_de_graaf_duration_ms=500, repeat=1):
    """Pollination sequence combining vibration and van de Graaf motors.
    
    Args:
        client_socket: Connected socket to Pi
        vibrate_duration_ms: Duration to vibrate in milliseconds
        van_de_graaf_duration_ms: Duration for van de Graaf in milliseconds
        repeat: Number of times to repeat the pollination cycle
    """
    print(f"[POLLINATE] Starting pollination sequence (x{repeat})")
    for cycle in range(repeat):
        print(f"[POLLINATE] Cycle {cycle+1}/{repeat}")
        # Vibrate first
        vibrate_motor(client_socket, vibrate_duration_ms)
        time.sleep(vibrate_duration_ms / 1000.0 + 0.1)  # Wait for vibration to complete
        # Then run van de Graaf
        van_de_graaf_motor(client_socket, van_de_graaf_duration_ms)
        time.sleep(van_de_graaf_duration_ms / 1000.0 + 0.1)  # Wait for van de Graaf to complete
    print(f"[POLLINATE] Pollination sequence complete")

# =============================
# Frame Reception Thread
# =============================
def frame_reception_thread():
    """Continuously receive frames from Pi."""
    global current_frame_left, current_frame_right, stereo_initialized, mapLx, mapLy, mapRx, mapRy, Q, K_L_use, frame_thread_failed, shutdown_flag
    
    data = b""
    payload_size = struct.calcsize("Q")
    
    try:
        while not shutdown_flag:
            frame_left, frame_right, data = receive_stereo_frames(client_socket, data, payload_size)
            
            if frame_left is None or frame_right is None:
                print("[Laptop] Failed to receive frames")
                frame_thread_failed = True
                break
            
            # Initialize stereo on first frame
            if not stereo_initialized:
                h, w = frame_left.shape[:2]
                w_rect = int(w * SCALE_FOR_MATCHING)
                h_rect = int(h * SCALE_FOR_MATCHING)
                mapLx, mapLy, mapRx, mapRy, Q, K_L_use = setup_stereo_rectification(K_L, D_L, K_R, D_R, R, T, w_rect, h_rect, SCALE_FOR_MATCHING)
                stereo_initialized = True
                print(f"[Laptop] Stereo initialized for {w}x{h} -> {w_rect}x{h_rect}")
            
            with frame_lock:
                current_frame_left = frame_left.copy()
                current_frame_right = frame_right.copy()
    
    except Exception as e:
        print(f"[Laptop] Frame reception error: {e}")
        frame_thread_failed = True

# =============================
# Demo Mode
# =============================
def demo_mode_worker():
    """Automatic flower detection and tracking demo."""
    global demo_stop_flag, current_frame_left, current_frame_right, demo_mode, OFFSET_X_CM, OFFSET_Y_CM, DEBUG_MOVEMENT
    
    # Reset position to origin at demo start
    reset_position(position_lock, current_position)
    print("[DEMO] Position reset to origin (0, 0, 0) - rear-right")
    print_position(position_lock, current_position)
    
    # Initialize arm: release and move up 50cm
    print("[DEMO] Initializing arm: releasing and moving up 50cm...")
    arm_release_steps = int(50 * STEPS_PER_CM_ARM)  # Move up 50cm
    send_motor_command(client_socket, {'arm': arm_release_steps, '_hold_motors': ['arm']})
    time.sleep(10.0)  # Wait for arm to reach position
    
    frame_count = 0
    flower_found = False  # Track if we've located a flower
    consecutive_no_detect = 0  # Track consecutive frames without detection
    
    # Square search pattern waypoints: (Y, X) in cm
    # Start (0,0) → (15,45) → (0,45) → (15,0) → back to (0,0)
    search_waypoints = [
        (15, 45),   # Front-left corner
        (0, 45),    # Front-right corner
        (15, 0),    # Rear-left corner
        (0, 0),     # Rear-right corner (start)
    ]
    current_waypoint_idx = 0
    target_waypoint = search_waypoints[current_waypoint_idx]
    
    saved_position = None  # Position before moving toward flower
    
    print("\n[DEMO] Starting automatic flower tracking demo...")
    print("[DEMO] Square search pattern: (0,0) → (15,45) → (0,45) → (15,0) → (0,0)")
    print("[DEMO] Press Ctrl+C to stop demo mode\n")
    
    try:
        while not demo_stop_flag and demo_mode:
            with frame_lock:
                if current_frame_left is None or current_frame_right is None:
                    time.sleep(0.05)
                    continue
                
                frame_left = current_frame_left.copy()
                frame_right = current_frame_right.copy()
            
            h, w = frame_left.shape[:2]
            w_rect = int(w * SCALE_FOR_MATCHING)
            h_rect = int(h * SCALE_FOR_MATCHING)
            
            frame_l_scaled = cv2.resize(frame_left, (w_rect, h_rect))
            frame_r_scaled = cv2.resize(frame_right, (w_rect, h_rect))
            
            rectL = cv2.remap(frame_l_scaled, mapLx, mapLy, cv2.INTER_LINEAR)
            rectR = cv2.remap(frame_r_scaled, mapRx, mapRy, cv2.INTER_LINEAR)
            
            disparity = compute_stereo_disparity(rectL, rectR, K_L_use, baseline, EXPECTED_DISTANCE)
            
            if disparity is None:
                print(f"[DEMO] Frame {frame_count}: Disparity failed, skipping")
                send_motor_command(client_socket, {})
                time.sleep(0.05)
                continue
            
            depth_map = compute_depth_map(disparity, Q)
            
            detections = detect_flowers(frame_left, model, device, YOLO_CONF, CONFIDENCE_THRESHOLD)
            
            depth_stats_list = []
            for det in detections:
                if DEBUG_SKIP_DEPTH:
                    depth_stats = None  # Skip depth estimation
                else:
                    scaled_box = tuple(int(coord * SCALE_FOR_MATCHING) for coord in det['box'])
                    depth_stats = estimate_roi_depth(depth_map, scaled_box)
                depth_stats_list.append(depth_stats)
            
            target_det, target_depth = select_target_flower(detections, depth_stats_list, w, h)
            
            motor_command = {}
            
            if target_det is None:
                consecutive_no_detect += 1
                # If no detection for 5 frames, reset flower_found to search mode
                if consecutive_no_detect > 5:
                    flower_found = False
                
                if not flower_found:
                    # Execute waypoint-based square search pattern
                    pos = get_current_position(position_lock, current_position)
                    target_y, target_x = target_waypoint
                    
                    # Move toward current waypoint
                    dx_to_waypoint = target_x - pos["x"]
                    dy_to_waypoint = target_y - pos["y"]
                    
                    # If reached waypoint (within 1cm tolerance), move to next
                    if abs(dx_to_waypoint) < 1.0 and abs(dy_to_waypoint) < 1.0:
                        current_waypoint_idx = (current_waypoint_idx + 1) % len(search_waypoints)
                        target_waypoint = search_waypoints[current_waypoint_idx]
                        target_y, target_x = target_waypoint
                        dx_to_waypoint = target_x - pos["x"]
                        dy_to_waypoint = target_y - pos["y"]
                        print(f"[DEMO] Frame {frame_count}: Reached waypoint, moving to next ({target_y:.1f}, {target_x:.1f})")
                    
                    # Move toward waypoint (small steps each frame)
                    move_step_cm = 1.0
                    if abs(dx_to_waypoint) > move_step_cm or abs(dy_to_waypoint) > move_step_cm:
                        # Normalize direction
                        dist = np.sqrt(dx_to_waypoint**2 + dy_to_waypoint**2)
                        dx_norm = (dx_to_waypoint / dist) * move_step_cm
                        dy_norm = (dy_to_waypoint / dist) * move_step_cm
                        
                        move_plan = convert_offsets_to_motor_steps(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, dx_norm * PIXELS_PER_CM, dy_norm * PIXELS_PER_CM, PIXELS_PER_CM, MAX_CM_PER_CYCLE, MAX_STEPS_PER_CYCLE, STEPS_PER_CM, scale_move, RAILS_CALIBRATION, MAIN_CALIBRATION, direction_dict)
                        motor_command = move_plan if move_plan else {}
                        if motor_command:
                            print(f"[DEMO] Frame {frame_count}: WAYPOINT SEARCH - Sending motor_command={motor_command}")
                            send_motor_command(client_socket, motor_command)
                else:
                    print(f"[DEMO] Frame {frame_count}: Lost flower (no detect {consecutive_no_detect}/5), moving back to waypoint...")
                    # Move back toward last saved waypoint position
                    pos = get_current_position(position_lock, current_position)
                    if saved_position:
                        dx_to_saved = saved_position["x"] - pos["x"]
                        dy_to_saved = saved_position["y"] - pos["y"]
                        move_plan = convert_offsets_to_motor_steps(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, dx_to_saved * PIXELS_PER_CM, dy_to_saved * PIXELS_PER_CM, PIXELS_PER_CM, MAX_CM_PER_CYCLE, MAX_STEPS_PER_CYCLE, STEPS_PER_CM, scale_move, RAILS_CALIBRATION, MAIN_CALIBRATION, direction_dict)
                        motor_command = move_plan if move_plan else {}
                        if motor_command:
                            print(f"[DEMO] Frame {frame_count}: RETURN TO SAVED - Sending motor_command={motor_command}")
                            send_motor_command(client_socket, motor_command)
                    else:
                        motor_command = {}
                    
                    # Send the movement command for waypoint search
                    if motor_command:
                        print(f"[DEMO] Frame {frame_count}: EXECUTING RETURN COMMAND: {motor_command}")
                        send_motor_command(client_socket, motor_command)
            else:
                consecutive_no_detect = 0  # Reset counter when we detect
                flower_found = True
                
                x1, y1, x2, y2 = target_det['box']
                flower_center_x = (x1 + x2) / 2
                flower_center_y = (y1 + y2) / 2
                
                dx = flower_center_x - (w / 2)
                dy = flower_center_y - (h / 2)
                depth_m = target_depth['median']
                
                # Calculate movement needed
                dx_cm_raw = clamp(dx / PIXELS_PER_CM, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE) + OFFSET_X_CM
                dy_cm_raw = clamp(dy / PIXELS_PER_CM, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE) + OFFSET_Y_CM
                dx_cm_allowed, dy_cm_allowed, _ = clamp_movement_to_limits(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, dx_cm_raw, dy_cm_raw, 0.0)
                
                # Check if this flower has already been visited
                pos = get_current_position(position_lock, current_position)
                flower_world_x = pos["x"] + dx_cm_allowed
                flower_world_y = pos["y"] + dy_cm_allowed
                
                if is_flower_already_visited(visited_flowers, flower_world_x, flower_world_y, FLOWER_VISIT_RADIUS_CM):
                    print(f"[DEMO] Frame {frame_count}: Flower already visited at ({flower_world_x:.2f}, {flower_world_y:.2f}) — ignoring")
                    target_det = None
                    target_depth = None
                    continue
                
                move_plan_preview = {}
                if abs(dx_cm_allowed) >= 0.1:
                    entries = direction_dict["left"] if dx_cm_allowed > 0 else direction_dict["right"]
                    steps_for_cm = abs(dx_cm_allowed) * STEPS_PER_CM * scale_move
                    for motor_name, multiplier in entries:
                        move_plan_preview[motor_name] = move_plan_preview.get(motor_name, 0) + int(multiplier * steps_for_cm)
                if abs(dy_cm_allowed) >= 0.1:
                    entries = direction_dict["rear"] if dy_cm_allowed > 0 else direction_dict["front"]
                    steps_for_cm = abs(dy_cm_allowed) * STEPS_PER_CM * scale_move
                    for motor_name, multiplier in entries:
                        move_plan_preview[motor_name] = move_plan_preview.get(motor_name, 0) + int(multiplier * steps_for_cm)
                rails_steps_log = move_plan_preview.get("rails", 0)
                main_steps_log = move_plan_preview.get("main", 0)
                
                if DEBUG_MOVEMENT:
                    # Debug mode: show visual and stop for inspection
                    # Display debug visualization
                    if SHOW_DEBUG:
                        display_frame_left = frame_left.copy()
                        display_frame_right = frame_right.copy()
                        
                        for det, depth_stats in zip(detections, depth_stats_list):
                            x1, y1, x2, y2 = det['box']
                            is_target = (det == target_det)
                            color = (0, 255, 0) if is_target else (128, 128, 128)
                            thickness = 3 if is_target else 1
                            
                            cv2.rectangle(display_frame_left, (x1, y1), (x2, y2), color, thickness)
                            
                            if depth_stats:
                                label = f"{det['confidence']:.2f} | {depth_stats['median']:.2f}m"
                                cv2.putText(display_frame_left, label, (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        cv2.line(display_frame_left, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 0, 0), 2)
                        cv2.line(display_frame_left, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 0, 0), 2)
                        
                        if target_det:
                            x1, y1, x2, y2 = target_det['box']
                            target_x = int((x1 + x2) / 2)
                            target_y = int((y1 + y2) / 2)
                            cv2.line(display_frame_left, (w//2, h//2), (target_x, target_y), (0, 255, 255), 2)
                            cv2.circle(display_frame_left, (target_x, target_y), 10, (0, 255, 0), -1)
                        
                        # Draw arm tip position marker (-130px left, +150px down from center)
                        arm_tip_x = int(w//2 - 130)
                        arm_tip_y = int(h//2 + 150)
                        if 0 <= arm_tip_x < w and 0 <= arm_tip_y < h:
                            cv2.circle(display_frame_left, (arm_tip_x, arm_tip_y), 8, (200, 0, 200), 2)  # Magenta circle for arm tip
                            cv2.putText(display_frame_left, "ARM", (arm_tip_x-15, arm_tip_y-15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 2)
                            # Draw line from arm tip to flower center
                            if target_det:
                                cv2.line(display_frame_left, (arm_tip_x, arm_tip_y), (target_x, target_y), (100, 200, 255), 2)
                        
                        # Add debug info to display
                        pos = get_current_position(position_lock, current_position)
                        # Line 1: Flower detected at this position
                        cv2.putText(display_frame_left, f"FLOWER: dx={dx:.0f}px, dy={dy:.0f}px", (10, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        # Line 2: Required movement without arm offset
                        cv2.putText(display_frame_left, f"  -> Move: rails={dx/PIXELS_PER_CM * STEPS_PER_CM * scale_move * RAILS_CALIBRATION:.0f}st, main={dy/PIXELS_PER_CM * STEPS_PER_CM * scale_move * MAIN_CALIBRATION:.0f}st", (10, 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        # Line 3: Arm tip offset
                        cv2.putText(display_frame_left, f"ARM OFFSET: dx={OFFSET_X_CM:+.2f}cm ({-130}px), dy={OFFSET_Y_CM:+.2f}cm ({+150}px)", (10, 110),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 200), 2)
                        # Line 4: Combined movement to position arm tip on flower
                        cv2.putText(display_frame_left, f"  -> Move: rails={rails_steps_log}st, main={main_steps_log}st", (10, 140),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
                        # Line 5: Final combined cm values
                        cv2.putText(display_frame_left, f"FINAL: dx={dx_cm_allowed:.2f}cm, dy={dy_cm_allowed:.2f}cm", (10, 170),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(display_frame_left, "[PAUSED] Press 'q' to continue...", (10, 200),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        # Apply rotation if configured
                        display_frame_left = rotate_frame(display_frame_left, ROTATE_LEFT)
                        display_frame_right = rotate_frame(display_frame_right, ROTATE_RIGHT)
                        
                        h_display = DISPLAY_HEIGHT // 2
                        w_display = DISPLAY_WIDTH // 2
                        display_frame_left = cv2.resize(display_frame_left, (w_display, h_display))
                        display_frame_right = cv2.resize(display_frame_right, (w_display, h_display))
                        
                        display_combined = np.hstack([display_frame_right, display_frame_left])
                        cv2.imshow("DEMO MODE - Automatic Flower Tracking", display_combined)
                    
                    # Print debug info
                    pos = get_current_position(position_lock, current_position)
                    print(f"\n[DEMO] FLOWER DETECTED - DEBUG MODE")
                    print(f"[DEMO] Frame {frame_count}: Confidence={target_det['confidence']:.2f}, Depth={depth_m:.3f}m")
                    print(f"[DEMO] Current Position: X={pos['x']:.2f}cm, Y={pos['y']:.2f}cm, Z={pos['z']:.2f}cm")
                    print(f"\n[DEMO] FLOWER (camera center to flower):")
                    print(f"[DEMO]   Pixels: dx={dx:.0f}px, dy={dy:.0f}px")
                    print(f"[DEMO]   CM: dx={dx/PIXELS_PER_CM:.2f}cm, dy={dy/PIXELS_PER_CM:.2f}cm")
                    flower_rails_steps = dx/PIXELS_PER_CM * STEPS_PER_CM * scale_move * RAILS_CALIBRATION
                    flower_main_steps = dy/PIXELS_PER_CM * STEPS_PER_CM * scale_move * MAIN_CALIBRATION
                    print(f"[DEMO]   Steps: rails={flower_rails_steps:.0f}, main={flower_main_steps:.0f}")
                    print(f"\n[DEMO] ARM TIP OFFSET (arm tip is -130px left, +150px down from camera):")
                    print(f"[DEMO]   Offset: dx={OFFSET_X_CM:+.2f}cm, dy={OFFSET_Y_CM:+.2f}cm")
                    print(f"\n[DEMO] FINAL (camera center + offset to position arm tip on flower):")
                    print(f"[DEMO]   CM: dx={dx_cm_allowed:.2f}cm, dy={dy_cm_allowed:.2f}cm")
                    print(f"[DEMO]   Steps: rails={rails_steps_log}, main={main_steps_log}")
                    print(f"[DEMO] *** PAUSED - Press 'q' in display window to continue ***\n")
                    
                    # Wait for 'q' key press while displaying
                    while True:
                        key = cv2.waitKey(100) & 0xFF
                        if key == ord('q') or key == 27:
                            DEBUG_MOVEMENT = False
                            print(f"[DEMO] Resuming from debug pause...\n")
                            break
                    continue
                else:
                    # Normal mode: approach flower
                    # Save current position before moving toward flower
                    if saved_position is None:
                        saved_position = get_current_position(position_lock, current_position).copy()
                        print(f"[DEMO] Position saved: ({saved_position['y']:.2f}, {saved_position['x']:.2f})")
                    
                    print(f"[DEMO] Frame {frame_count}: Flower found! dx={dx:.0f}px, dy={dy:.0f}px, depth={depth_m:.3f}m (conf={target_det['confidence']:.2f})")
                    print(f"[DEMO] Movement: dx={dx_cm_allowed:.2f}cm, dy={dy_cm_allowed:.2f}cm | rails={rails_steps_log} steps, main={main_steps_log} steps")

                    # Execute a small movement toward the flower center
                    move_plan = convert_offsets_to_motor_steps(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, dx, dy, PIXELS_PER_CM, MAX_CM_PER_CYCLE, MAX_STEPS_PER_CYCLE, STEPS_PER_CM, scale_move, RAILS_CALIBRATION, MAIN_CALIBRATION, direction_dict)
                    if move_plan:
                        print(f"[DEMO] Frame {frame_count}: FLOWER APPROACH - Sending motor_command={move_plan}")
                        send_motor_command(client_socket, move_plan)
                        time.sleep(0.3)  # Short wait for movement to complete
                    else:
                        # If no movement needed (already centered), proceed to arm down
                        print(f"[DEMO] Already centered on flower, proceeding with arm down")
                    
                    # Always move arm down 30cm when flower detected
                    arm_steps = int(-32 * STEPS_PER_CM_ARM)  # Move down 32cm
                    arm_command = {'arm': arm_steps, '_hold_motors': ['arm']}  # Keep arm powered to hold position
                    print(f"[DEMO] Frame {frame_count}: ARM DOWN - Sending motor_command={arm_command}")
                    
                    send_motor_command(client_socket, arm_command)
                    time.sleep(7)  # Wait for arm to reach flower
                    
                    # Pollinate the flower
                    pollinate(client_socket, vibrate_duration_ms=500, van_de_graaf_duration_ms=500, repeat=1)
                    
                    # Mark flower as visited at its world coordinates
                    mark_flower_as_visited(visited_flowers, flower_world_x, flower_world_y)
                    
                    # Move arm back up 32cm
                    arm_up_steps = int(32 * STEPS_PER_CM_ARM)  # Move up 32cm
                    arm_command_up = {'arm': arm_up_steps, '_hold_motors': ['arm']}  # Keep arm held after retract
                    print(f"[DEMO] Frame {frame_count}: ARM UP - Sending motor_command={arm_command_up}")
                    send_motor_command(client_socket, arm_command_up)
                    print(f"[DEMO] Arm retracted: {arm_up_steps} steps (moving up 32cm)")
                    time.sleep(7)  # Wait for arm to retract
                    
                    # Move back to saved position
                    if saved_position:
                        pos = get_current_position(position_lock, current_position)
                        dx_to_saved = saved_position["x"] - pos["x"]
                        dy_to_saved = saved_position["y"] - pos["y"]
                        print(f"[DEMO] Moving back to saved position: ({saved_position['y']:.2f}, {saved_position['x']:.2f})")
                        move_plan = convert_offsets_to_motor_steps(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, dx_to_saved * PIXELS_PER_CM, dy_to_saved * PIXELS_PER_CM, PIXELS_PER_CM, MAX_CM_PER_CYCLE, MAX_STEPS_PER_CYCLE, STEPS_PER_CM, scale_move, RAILS_CALIBRATION, MAIN_CALIBRATION, direction_dict)
                        if move_plan:
                            print(f"[DEMO] Frame {frame_count}: RETURN TO SAVED POSITION - Sending motor_command={move_plan}")
                            send_motor_command(client_socket, move_plan)
                            time.sleep(7)  # Wait for movement to complete
                        saved_position = None  # Clear saved position for next flower
                    
                    # Don't send empty command at the end of frame when flower detected
                    time.sleep(0.05)
            
            # Display (runs every frame, not skipped by continue)
            if SHOW_DEBUG:
                display_frame_left = frame_left.copy()
                display_frame_right = frame_right.copy()
                
                for det, depth_stats in zip(detections, depth_stats_list):
                    x1, y1, x2, y2 = det['box']
                    is_target = (det == target_det)
                    color = (0, 255, 0) if is_target else (128, 128, 128)
                    thickness = 3 if is_target else 1
                    
                    cv2.rectangle(display_frame_left, (x1, y1), (x2, y2), color, thickness)
                    
                    if depth_stats:
                        label = f"{det['confidence']:.2f} | {depth_stats['median']:.2f}m"
                        cv2.putText(display_frame_left, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.line(display_frame_left, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 0, 0), 2)
                cv2.line(display_frame_left, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 0, 0), 2)
                
                if target_det:
                    x1, y1, x2, y2 = target_det['box']
                    target_x = int((x1 + x2) / 2)
                    target_y = int((y1 + y2) / 2)
                    cv2.line(display_frame_left, (w//2, h//2), (target_x, target_y), (0, 255, 255), 2)
                    cv2.circle(display_frame_left, (target_x, target_y), 10, (0, 255, 0), -1)
                
                cv2.putText(display_frame_left, f"[DEMO] Frame: {frame_count} | Detections: {len(detections)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame_right, "[DEMO] Right Camera (Reference)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Apply rotation if configured
                display_frame_left = rotate_frame(display_frame_left, ROTATE_LEFT)
                display_frame_right = rotate_frame(display_frame_right, ROTATE_RIGHT)
                
                h_display = DISPLAY_HEIGHT // 2
                w_display = DISPLAY_WIDTH // 2
                display_frame_left = cv2.resize(display_frame_left, (w_display, h_display))
                display_frame_right = cv2.resize(display_frame_right, (w_display, h_display))
                
                display_combined = np.hstack([display_frame_right, display_frame_left])
                cv2.imshow("DEMO MODE - Automatic Flower Tracking", display_combined)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    demo_stop_flag = True
                    break
            
            frame_count += 1
            time.sleep(0.05)
    
    except Exception as e:
        print(f"[DEMO] Error: {e}")
    finally:
        print("[DEMO] Demo mode stopped")
        send_motor_command(client_socket, {})
        demo_mode = False

# =============================
# Keyboard Control Mode
# =============================
def keyboard_control_mode():
    """Real-time keyboard control with immediate response to key press/release.
    
    Controls:
    - W: Move forward (main +) - hold for continuous movement
    - S: Move backward (main -) - hold for continuous movement
    - A: Move left (rails +) - hold for continuous movement
    - D: Move right (rails -) - hold for continuous movement
    - O: Move arm up 1cm - single press (not continuous)
    - P: Pollinate sequence - single press (down 32cm -> VDG 2s -> up 32cm)
    - ESC: Exit keyboard mode
    """
    global shutdown_flag, current_frame_left, current_frame_right
    
    print("\n[KEYBOARD] Entering keyboard control mode")
    print("[KEYBOARD] Controls:")
    print("[KEYBOARD]   W: Forward (main +) - hold for continuous")
    print("[KEYBOARD]   S: Backward (main -) - hold for continuous")
    print("[KEYBOARD]   A: Left (rails +) - hold for continuous")
    print("[KEYBOARD]   D: Right (rails -) - hold for continuous")
    print("[KEYBOARD]   O: Arm up 1cm - single press")
    print("[KEYBOARD]   P: Pollinate sequence - single press")
    print("[KEYBOARD]   ESC: Exit")
    print("[KEYBOARD] Initializing arm: moving up 50cm...\n")
    
    # Initialize arm: move up 50cm at start
    arm_init_steps = int(50 * STEPS_PER_CM_ARM)
    send_motor_command(client_socket, {'arm': arm_init_steps, '_hold_motors': ['arm']})
    print("[KEYBOARD] Arm initialization sent (50cm up)\n")
    time.sleep(5.0)  # Wait for arm to reach position
    
    window_name = "KEYBOARD CONTROL MODE - Hold keys for continuous movement (ESC to exit)"
    step_cm = 1.0  # Movement per frame in cm
    arm_step_cm = 1.0  # Arm movement per single press in cm
    
    # Key state tracking - using pynput for async key detection
    key_states = {
        'w': False, 'W': False,
        's': False, 'S': False,
        'a': False, 'A': False,
        'd': False, 'D': False,
    }
    
    # Track single-press keys separately
    arm_up_pressed = False
    arm_up_processed = False
    pollinate_pressed = False
    pollinate_processed = False
    
    exit_flag = False
    
    def on_press(key):
        """Handle key press."""
        nonlocal arm_up_pressed, pollinate_pressed
        try:
            key_char = key.char if hasattr(key, 'char') else None
            if key_char and key_char.lower() in ['w', 's', 'a', 'd']:
                key_states[key_char] = True
            elif key_char and key_char.lower() == 'o':
                arm_up_pressed = True
                print(f"[KEYBOARD] Key pressed: O (arm up)")
            elif key_char and key_char.lower() == 'p':
                pollinate_pressed = True
                print(f"[KEYBOARD] Key pressed: P (pollinate)")
        except:
            pass
    
    def on_release(key):
        """Handle key release."""
        nonlocal exit_flag, arm_up_pressed, arm_up_processed, pollinate_pressed, pollinate_processed
        try:
            # Check for ESC key to exit
            if key == keyboard.Key.esc:
                exit_flag = True
                return False  # Stop listener
            
            key_char = key.char if hasattr(key, 'char') else None
            if key_char and key_char.lower() in ['w', 's', 'a', 'd']:
                key_states[key_char] = False
            elif key_char and key_char.lower() == 'o':
                arm_up_pressed = False
                arm_up_processed = False  # Allow next press
                print(f"[KEYBOARD] Key released: O")
            elif key_char and key_char.lower() == 'p':
                pollinate_pressed = False
                pollinate_processed = False  # Allow next press
                print(f"[KEYBOARD] Key released: P")
        except:
            pass
    
    # Start keyboard listener in background
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("[KEYBOARD] Keyboard listener started (async)\n")
    
    try:
        while not shutdown_flag and not exit_flag:
            # Get current frame for display
            with frame_lock:
                if current_frame_left is None:
                    time.sleep(0.01)
                    continue
                display_frame = current_frame_left.copy()
            
            # Apply rotation if configured
            display_frame = rotate_frame(display_frame, ROTATE_LEFT)
            
            # Add info text
            pos = get_current_position(position_lock, current_position)
            cv2.putText(display_frame, "KEYBOARD CONTROL MODE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Position: X={pos['x']:.1f}cm Y={pos['y']:.1f}cm Z={pos['z']:.1f}cm", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Show which keys are currently pressed
            active_keys = [k.upper() for k, v in key_states.items() if v]
            if active_keys:
                cv2.putText(display_frame, f"Active keys: {' '.join(set(active_keys))}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
            
            cv2.putText(display_frame, "W-Forward  S-Backward  A-Left  D-Right  O-ArmUp  P-Pollinate  ESC-Exit", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            
            h_display = DISPLAY_HEIGHT // 2
            w_display = DISPLAY_WIDTH // 2
            display_frame = cv2.resize(display_frame, (w_display, h_display))
            
            cv2.imshow(window_name, display_frame)
            
            # Non-blocking display (10ms wait for window events)
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC in OpenCV window
                exit_flag = True
                break
            
            # Process continuous movement based on held keys (W/A/S/D only)
            motor_command = {}
            
            rails_movement = 0
            main_movement = 0
            
            # Rails control (A/D)
            if key_states['a'] or key_states['A']:  # Left (rails increase)
                steps = int(step_cm * STEPS_PER_CM * RAILS_CALIBRATION)
                rails_movement += steps
                update_position(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, delta_x=step_cm)
            
            if key_states['d'] or key_states['D']:  # Right (rails decrease)
                steps = int(step_cm * STEPS_PER_CM * RAILS_CALIBRATION)
                rails_movement -= steps
                update_position(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, delta_x=-step_cm)
            
            # Main control (W/S)
            if key_states['w'] or key_states['W']:  # Forward (main increase)
                steps = int(step_cm * STEPS_PER_CM * MAIN_CALIBRATION)
                main_movement += steps
                update_position(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, delta_y=step_cm)
            
            if key_states['s'] or key_states['S']:  # Backward (main decrease)
                steps = int(step_cm * STEPS_PER_CM * MAIN_CALIBRATION)
                main_movement -= steps
                update_position(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, delta_y=-step_cm)
            
            # Build and send motor command for continuous movements
            if rails_movement != 0:
                motor_command['rails'] = rails_movement
            
            if main_movement != 0:
                motor_command['main'] = main_movement
            
            if motor_command:
                send_motor_command(client_socket, motor_command)
            
            # Process single-press keys (O for arm up, P for pollinate)
            if arm_up_pressed and not arm_up_processed:
                # Execute arm up movement
                arm_steps = int(arm_step_cm * STEPS_PER_CM_ARM)
                arm_command = {'arm': arm_steps, '_hold_motors': ['arm']}
                print(f"[KEYBOARD] Moving ARM UP 1cm")
                send_motor_command(client_socket, arm_command)
                update_position(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, delta_z=arm_step_cm)
                arm_up_processed = True
            
            if pollinate_pressed and not pollinate_processed:
                # Execute pollination sequence
                print(f"[KEYBOARD] Executing pollination sequence...")
                print(f"[KEYBOARD]   1. Moving arm DOWN 32cm")
                arm_down_steps = int(-32 * STEPS_PER_CM_ARM)
                send_motor_command(client_socket, {'arm': arm_down_steps, '_hold_motors': ['arm']})
                time.sleep(5.0)  # Wait for arm to reach flower
                
                print(f"[KEYBOARD]   2. Running VDG for 2 seconds")
                van_de_graaf_motor(client_socket, 2000)
                time.sleep(2.5)  # Wait for VDG to complete
                
                print(f"[KEYBOARD]   3. Moving arm UP 32cm")
                arm_up_steps = int(32 * STEPS_PER_CM_ARM)
                send_motor_command(client_socket, {'arm': arm_up_steps, '_hold_motors': ['arm']})
                time.sleep(5.0)  # Wait for arm to retract
                
                print(f"[KEYBOARD] Pollination sequence complete")
                pollinate_processed = True
            
            time.sleep(0.5)
    
    except Exception as e:
        print(f"[KEYBOARD] Error: {e}")
    finally:
        listener.stop()
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
        print("[KEYBOARD] Keyboard control mode stopped")

# =============================
# Command Interface
# =============================
def parse_and_execute_command(cmd_input):
    """Parse user command and execute."""
    global demo_mode, demo_stop_flag, arm_is_at_bottom
    
    cmd_input = cmd_input.strip().lower()
    
    if not cmd_input:
        return
    
    parts = cmd_input.split()
    command = parts[0]
    
    try:
        if command == "status":
            print_position(position_lock, current_position)
        
        elif command == "reset":
            print("[COMMAND] Resetting all motors to home position...")
            motor_command = {"rails": 0, "main": 0, "arm": 0}
            send_motor_command(client_socket, motor_command)
            reset_position(position_lock, current_position)
            arm_is_at_bottom = True
            print("[COMMAND] Reset command sent! Position reset to origin (0, 0, 0)")
            print_position(position_lock, current_position)
        
        elif command == "move":
            if len(parts) < 3:
                print("[ERROR] Format: move <motor> <steps>")
                print("[ERROR] Example: move rails 100")
                return
            
            motor_name = parts[1]
            if motor_name not in ["rails", "main", "arm"]:
                print(f"[ERROR] Unknown motor: {motor_name}")
                print("[ERROR] Valid motors: rails, main, arm")
                return
            
            try:
                steps = int(parts[2])
            except ValueError:
                print(f"[ERROR] Invalid steps value: {parts[2]}")
                return
            
            motor_command = {motor_name: steps}
            print(f"[COMMAND] Moving {motor_name} for {steps} steps...")
            send_motor_command(client_socket, motor_command)
            print("[COMMAND] Move command sent!")

        elif command == "movecm":
            if len(parts) < 3:
                print("[ERROR] Format: movecm <motor> <cm>")
                print("[ERROR] Example: movecm rails 2")
                return

            motor_name = parts[1]
            if motor_name not in ["rails", "main", "arm"]:
                print(f"[ERROR] movecm supports rails/main/arm (got {motor_name})")
                return

            try:
                dist_cm = float(parts[2])
            except ValueError:
                print(f"[ERROR] Invalid cm value: {parts[2]}")
                return

            # Enforce boundary limits before converting to steps
            if motor_name == "rails":
                dist_cm, _, _ = clamp_movement_to_limits(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, dist_cm, 0.0, 0.0)
            elif motor_name == "main":
                _, dist_cm, _ = clamp_movement_to_limits(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, 0.0, dist_cm, 0.0)
            elif motor_name == "arm":
                if arm_is_at_bottom and dist_cm > 0:
                    adjusted_dist_cm = dist_cm + ARM_BOTTOM_OFFSET_CM
                    print(f"[COMMAND] Arm at bottom, adding {ARM_BOTTOM_OFFSET_CM} cm offset")
                    print(f"[COMMAND] Adjusted distance: {dist_cm:.2f} cm -> {adjusted_dist_cm:.2f} cm")
                    dist_cm = adjusted_dist_cm
                    arm_is_at_bottom = False
                elif dist_cm < 0:
                    arm_is_at_bottom = False
                
                _, _, dist_cm = clamp_movement_to_limits(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, 0.0, 0.0, dist_cm)

            # Use appropriate calibration for each motor
            if motor_name == "arm":
                steps = int(dist_cm * STEPS_PER_CM_ARM)
            else:
                steps = int(dist_cm * STEPS_PER_CM)

            motor_command = {motor_name: steps}
            # Keep arm held after movement
            if motor_name == "arm" and steps != 0:
                motor_command["_hold_motors"] = ["arm"]
            
            print(f"[COMMAND] Moving {motor_name} ~{dist_cm:.2f} cm ({steps} steps)")
            send_motor_command(client_socket, motor_command)
            
            # Update position manually for direct motor commands
            if motor_name == "rails":
                update_position(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, delta_x=dist_cm)
            elif motor_name == "main":
                update_position(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, delta_y=dist_cm)
            elif motor_name == "arm":
                update_position(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, delta_z=dist_cm)
            
            print_position(position_lock, current_position)
            print("[COMMAND] Move command sent!")
        
        elif command == "arm":
            if len(parts) < 2:
                print("[ERROR] Format: arm <steps>")
                print("[ERROR] Example: arm 10")
                return
            
            try:
                steps = int(parts[1])
            except ValueError:
                print(f"[ERROR] Invalid steps value: {parts[1]}")
                return
            
            motor_command = {"arm": steps, "_hold_motors": ["arm"]}
            print(f"[COMMAND] Moving arm for {steps} steps (held)...")
            send_motor_command(client_socket, motor_command)
            
            # Update position
            delta_z = steps / STEPS_PER_CM_ARM
            update_position(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX, delta_z=delta_z)
            print_position(position_lock, current_position)
            print("[COMMAND] Arm command sent!")
        
        elif command == "release":
            print("[COMMAND] Releasing all motors...")
            motor_command = {"_action": "release_all"}
            send_motor_command(client_socket, motor_command)
            print("[COMMAND] Release command sent!")
        
        elif command == "demo":
            if demo_mode:
                print("[COMMAND] Demo mode already running!")
                return
            
            demo_stop_flag = False
            demo_mode = True
            demo_thread = threading.Thread(target=demo_mode_worker, daemon=True)
            demo_thread.start()
        
        elif command == "stop":
            if not demo_mode:
                print("[COMMAND] Demo mode is not running")
                return
            
            print("[COMMAND] Stopping demo mode...")
            motor_command = {"_action": "release_all"}
            send_motor_command(client_socket, motor_command)
            demo_stop_flag = True
            time.sleep(0.5)
            print("[COMMAND] Demo mode stopped")
            print_position(position_lock, current_position)
        
        elif command == "vibrate":
            if len(parts) < 2:
                print("[ERROR] Format: vibrate <milliseconds>")
                print("[ERROR] Example: vibrate 500")
                return
            
            try:
                duration_ms = int(parts[1])
                if duration_ms <= 0:
                    print("[ERROR] Duration must be positive")
                    return
            except ValueError:
                print(f"[ERROR] Invalid duration value: {parts[1]}")
                return
            
            vibrate_motor(client_socket, duration_ms)
            print("[COMMAND] Vibrate command sent!")
        
        elif command == "vdg":
            if len(parts) < 2:
                print("[ERROR] Format: vdg <milliseconds>")
                print("[ERROR] Example: vdg 500")
                return
            
            try:
                duration_ms = int(parts[1])
                if duration_ms <= 0:
                    print("[ERROR] Duration must be positive")
                    return
            except ValueError:
                print(f"[ERROR] Invalid duration value: {parts[1]}")
                return
            
            van_de_graaf_motor(client_socket, duration_ms)
            print("[COMMAND] Van de Graaf command sent!")
        
        elif command == "pollinate":
            # Parse optional parameters: pollinate [vibrate_ms] [van_de_graaf_ms] [repeat]
            vibrate_ms = 500
            van_de_graaf_ms = 500
            repeat = 1
            
            if len(parts) > 1:
                try:
                    vibrate_ms = int(parts[1])
                except ValueError:
                    print(f"[ERROR] Invalid vibrate duration: {parts[1]}")
                    return
            
            if len(parts) > 2:
                try:
                    van_de_graaf_ms = int(parts[2])
                except ValueError:
                    print(f"[ERROR] Invalid van_de_graaf duration: {parts[2]}")
                    return
            
            if len(parts) > 3:
                try:
                    repeat = int(parts[3])
                except ValueError:
                    print(f"[ERROR] Invalid repeat count: {parts[3]}")
                    return
            
            if vibrate_ms <= 0 or van_de_graaf_ms <= 0 or repeat <= 0:
                print("[ERROR] All parameters must be positive")
                return
            
            pollinate(client_socket, vibrate_ms, van_de_graaf_ms, repeat)
            print("[COMMAND] Pollinate command sent!")
        
        elif command == "help":
            print_help()
        
        elif command == "keys" or command == "keyboard":
            keyboard_control_mode()
        
        elif command == "quit" or command == "exit":
            return "QUIT"
        
        else:
            print(f"[ERROR] Unknown command: {command}")
            print("[ERROR] Type 'help' for available commands")
    
    except Exception as e:
        print(f"[ERROR] Command execution failed: {e}")

# =============================
# Main
# =============================
def main():
    global client_socket, shutdown_flag, frame_thread_failed, demo_mode, demo_stop_flag
    
    # Connect to Pi
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((PI_IP, PORT))
        print(f"[Laptop] Connected to Pi at {PI_IP}:{PORT}")
    except Exception as e:
        print(f"[Laptop] Connection failed: {e}")
        return
    
    # Start frame reception thread
    frame_thread = threading.Thread(target=frame_reception_thread, daemon=True)
    frame_thread.start()
    print("[Laptop] Frame reception thread started")
    
    # Wait for frames to arrive (longer, with error detection)
    print("[Laptop] Waiting for first frame from Pi...")
    wait_start = time.time()
    max_wait = 15  # seconds
    while current_frame_left is None and not frame_thread_failed:
        if time.time() - wait_start > max_wait:
            break
        time.sleep(0.5)
    
    if frame_thread_failed:
        print("[Laptop] Frame reception failed. Check Pi server and connection.")
        shutdown_flag = True
        client_socket.close()
        return
    
    if current_frame_left is None:
        print(f"[Laptop] No frames received from Pi within {max_wait}s. Exiting.")
        shutdown_flag = True
        client_socket.close()
        return
    
    print("[Laptop] Frames received! Ready for commands.")

    # Start display thread for manual viewing
    display_thread_obj = None
    if SHOW_DEBUG:
        display_thread_obj = threading.Thread(target=display_thread, daemon=True)
        display_thread_obj.start()
        print("[Laptop] Display thread started (manual view)")
    
    # Initialize position to origin
    reset_position(position_lock, current_position)
    print("\n[Laptop] Position initialized to origin (0, 0, 0) - rear-right")
    
    print_help()
    
    try:
        while True:
            try:
                cmd_input = input("> ").strip()
                result = parse_and_execute_command(cmd_input)
                if result == "QUIT":
                    break
            except KeyboardInterrupt:
                print("\n[Laptop] Interrupted by user")
                break
            except EOFError:
                break
    
    except KeyboardInterrupt:
        print("\n[Laptop] Interrupted by user")
    except Exception as e:
        print(f"[Laptop] Error: {e}")
    finally:
        demo_stop_flag = True
        demo_mode = False
        shutdown_flag = True
        time.sleep(0.5)
        
        client_socket.close()
        cv2.destroyAllWindows()
        print("[Laptop] Cleanup complete")

if __name__ == "__main__":
    main()
