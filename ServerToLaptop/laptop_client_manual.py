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
scale_move = 0.1

PIXELS_PER_CM = 10.0
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
    "front": [("main", 1), ("main", 1)],
    "rear": [("main", -1), ("main", -1)],
    "right": [("rails", 1), ("rails", 1)],
    "left": [("rails", -1), ("rails", -1)],
    "up": [("arm", 1)],
    "down": [("arm", -1)],
}


# Debug mode
DEBUG_MOVEMENT = False  # Set to True to stop at flower and display movement calculations
DEBUG_SKIP_DEPTH = True  # Set to True to skip depth estimation and detection

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
# Utility Functions
# =============================
def clamp(n, small, large):
    return max(small, min(n, large))

def get_current_position():
    """Get current position safely."""
    with position_lock:
        return current_position.copy()

def update_position(delta_x=0.0, delta_y=0.0, delta_z=0.0):
    """Update position and enforce limits."""
    with position_lock:
        new_x = current_position["x"] + delta_x
        new_y = current_position["y"] + delta_y
        new_z = current_position["z"] + delta_z
        
        # Clamp to limits
        current_position["x"] = clamp(new_x, LIMIT_X_MIN, LIMIT_X_MAX)
        current_position["y"] = clamp(new_y, LIMIT_Y_MIN, LIMIT_Y_MAX)
        current_position["z"] = clamp(new_z, LIMIT_Z_MIN, LIMIT_Z_MAX)

def reset_position():
    """Reset to origin (0,0,0) - rear-right."""
    with position_lock:
        current_position["x"] = 0.0
        current_position["y"] = 0.0
        current_position["z"] = 0.0

def print_position():
    """Print current position and limits."""
    pos = get_current_position()
    print("\n" + "="*60)
    print("CURRENT POSITION")
    print("="*60)
    print(f"  X (rails):  {pos['x']:6.2f} cm  [0.00 - 45.00] (0=right, 45=left)")
    print(f"  Y (main):   {pos['y']:6.2f} cm  [0.00 - 18.00] (0=rear, 18=front)")
    print(f"  Z (arm):    {pos['z']:6.2f} cm  [0.00 - 20.00] (0=down, 20=up)")
    print(f"\nOrigin (0,0,0) is rear-right corner")
    print("="*60 + "\n")

def clamp_movement_to_limits(delta_x_cm, delta_y_cm, delta_z_cm):
    """Clamp desired movement to stay within limits."""
    pos = get_current_position()
    
    # Calculate final positions if movement applied
    final_x = pos["x"] + delta_x_cm
    final_y = pos["y"] + delta_y_cm
    final_z = pos["z"] + delta_z_cm
    
    # Clamp and calculate actual allowed movement
    clamped_x = clamp(final_x, LIMIT_X_MIN, LIMIT_X_MAX)
    clamped_y = clamp(final_y, LIMIT_Y_MIN, LIMIT_Y_MAX)
    clamped_z = clamp(final_z, LIMIT_Z_MIN, LIMIT_Z_MAX)
    
    actual_delta_x = clamped_x - pos["x"]
    actual_delta_y = clamped_y - pos["y"]
    actual_delta_z = clamped_z - pos["z"]
    
    # Report if movement was limited
    if (abs(actual_delta_x - delta_x_cm) > 0.01 or 
        abs(actual_delta_y - delta_y_cm) > 0.01 or 
        abs(actual_delta_z - delta_z_cm) > 0.01):
        print(f"[BOUNDARY] Movement limited:")
        if abs(actual_delta_x - delta_x_cm) > 0.01:
            print(f"  X: {delta_x_cm:.2f}cm -> {actual_delta_x:.2f}cm")
        if abs(actual_delta_y - delta_y_cm) > 0.01:
            print(f"  Y: {delta_y_cm:.2f}cm -> {actual_delta_y:.2f}cm")
        if abs(actual_delta_z - delta_z_cm) > 0.01:
            print(f"  Z: {delta_z_cm:.2f}cm -> {actual_delta_z:.2f}cm")
    
    return actual_delta_x, actual_delta_y, actual_delta_z

def rotate_frame(frame, rotation_degrees):
    """Rotate frame by specified degrees (0, 90, 180, 270)."""
    if rotation_degrees == 0:
        return frame
    elif rotation_degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return frame

def scale_intrinsics(K, scale):
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale
    K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale
    K_scaled[1, 2] *= scale
    return K_scaled

def setup_stereo_rectification(w, h, scale):
    """Setup stereo rectification maps."""
    K_L_use = scale_intrinsics(K_L, scale)
    K_R_use = scale_intrinsics(K_R, scale)
    
    R1, R2, P1, P2, Q_out, _, _ = cv2.stereoRectify(
        K_L_use, D_L, K_R_use, D_R, (w, h), R, T, alpha=1
    )
    
    mapLx, mapLy = cv2.initUndistortRectifyMap(
        K_L_use, D_L, R1, P1, (w, h), cv2.CV_32FC1
    )
    mapRx, mapRy = cv2.initUndistortRectifyMap(
        K_R_use, D_R, R2, P2, (w, h), cv2.CV_32FC1
    )
    
    return mapLx, mapLy, mapRx, mapRy, Q_out, K_L_use

def compute_stereo_disparity(rectL, rectR, K_L_use):
    """Compute disparity map from rectified stereo pair."""
    if rectL.shape != rectR.shape:
        print(f"[WARNING] Shape mismatch: rectL={rectL.shape}, rectR={rectR.shape}")
        return None
    
    if rectL.ndim == 3:
        rectL_gray = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        rectR_gray = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    else:
        rectL_gray = rectL
        rectR_gray = rectR
    
    if rectL_gray.dtype != np.uint8:
        rectL_gray = rectL_gray.astype(np.uint8)
    if rectR_gray.dtype != np.uint8:
        rectR_gray = rectR_gray.astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    rectL_proc = clahe.apply(rectL_gray)
    rectR_proc = clahe.apply(rectR_gray)
    
    try:
        expected_disp = (K_L_use[0, 0] * baseline) / EXPECTED_DISTANCE
        num_disp = int(np.ceil(expected_disp * 1.8 / 16.0) * 16)
        num_disp = max(16, min(num_disp, 256))
        num_disp = (num_disp // 16) * 16
        if num_disp < 16:
            num_disp = 16
    except Exception as e:
        print(f"[WARNING] Disparity calculation error: {e}, using default 128")
        num_disp = 128
    
    h, w = rectL_proc.shape
    block_size = 5 if min(h, w) > 200 else 3
    
    stereo_matcher = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    try:
        disparity = stereo_matcher.compute(rectL_proc, rectR_proc).astype(np.float32) / 16.0
        return disparity
    except Exception as e:
        print(f"[ERROR] Stereo compute failed: {e}")
        return None

def compute_depth_map(disparity, Q):
    """Convert disparity to depth map."""
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    depth_map = points_3d[:, :, 2]
    
    invalid_mask = (disparity <= 0) | ~np.isfinite(depth_map)
    depth_map[invalid_mask] = 0.0
    
    return depth_map

def detect_flowers(frame, conf_threshold=YOLO_CONF):
    """Run YOLO detection on frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(source=rgb, conf=conf_threshold, device=device, verbose=False)
    
    detections = []
    if results and len(results) > 0:
        r = results[0]
        if hasattr(r, 'boxes') and r.boxes is not None:
            xyxy = r.boxes.xyxy
            confs = r.boxes.conf
            cls_ids = r.boxes.cls
            
            if hasattr(xyxy, 'cpu'):
                xyxy = xyxy.cpu().numpy()
            if hasattr(confs, 'cpu'):
                confs = confs.cpu().numpy()
            if hasattr(cls_ids, 'cpu'):
                cls_ids = cls_ids.cpu().numpy()
            
            if xyxy is not None and confs is not None:
                for i, box in enumerate(xyxy):
                    conf = float(confs[i])
                    if conf >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box[:4])
                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'confidence': conf,
                            'cls_id': int(cls_ids[i]) if cls_ids is not None else -1
                        })
    
    return detections

def estimate_roi_depth(depth_map, roi_box, scale_factor=1.0):
    """Estimate depth for bounding box ROI."""
    x1, y1, x2, y2 = roi_box
    
    if scale_factor != 1.0:
        x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)
    
    h, w = depth_map.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    roi_depth = depth_map[y1:y2, x1:x2]
    valid_depths = roi_depth[roi_depth > 0]
    
    if len(valid_depths) == 0:
        return None
    
    return {
        'median': float(np.median(valid_depths)),
        'mean': float(valid_depths.mean()),
        'valid_pixels': len(valid_depths),
        'coverage': 100.0 * len(valid_depths) / roi_depth.size
    }

def select_target_flower(detections, depth_stats_list, frame_width, frame_height):
    """Select one flower to track."""
    if not detections:
        return None, None
    
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    best_idx = None
    best_score = float('inf')
    
    for i, (det, depth_stats) in enumerate(zip(detections, depth_stats_list)):
        if depth_stats is None or depth_stats['valid_pixels'] == 0:
            continue
        
        x1, y1, x2, y2 = det['box']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        dist_from_center = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
        
        depth = depth_stats['median']
        score = depth * 0.01 + dist_from_center * 0.9
        
        if score < best_score:
            best_score = score
            best_idx = i
    
    if best_idx is not None:
        return detections[best_idx], depth_stats_list[best_idx]
    
    return None, None

def convert_offsets_to_motor_steps(dx_pixels, dy_pixels):
    """Convert pixel offsets to motor steps, respecting boundaries."""
    dx_cm = dx_pixels / PIXELS_PER_CM
    dy_cm = dy_pixels / PIXELS_PER_CM
    
    dx_cm = clamp(dx_cm, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE)
    dy_cm = clamp(dy_cm, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE)
    
    # Respect boundary limits
    dx_cm, dy_cm, _ = clamp_movement_to_limits(dx_cm, dy_cm, 0.0)
    
    move_plan = {}
    
    # Lower threshold to 0.1cm for smaller movements
    if abs(dx_cm) >= 0.1:
        if dx_cm > 0:
            entries = direction_dict["left"]
        else:
            entries = direction_dict["right"]
        steps_for_cm = abs(dx_cm) * STEPS_PER_CM * scale_move
        for motor_name, multiplier in entries:
            motor_steps = int(multiplier * steps_for_cm)
            move_plan[motor_name] = move_plan.get(motor_name, 0) + motor_steps
    
    if abs(dy_cm) >= 0.1:
        if dy_cm > 0:
            entries = direction_dict["rear"]
        else:
            entries = direction_dict["front"]
        steps_for_cm = abs(dy_cm) * STEPS_PER_CM * scale_move
        for motor_name, multiplier in entries:
            motor_steps = int(multiplier * steps_for_cm)
            move_plan[motor_name] = move_plan.get(motor_name, 0) + motor_steps
    
    for k in list(move_plan.keys()):
        capped = clamp(move_plan[k], -MAX_STEPS_PER_CYCLE, MAX_STEPS_PER_CYCLE)
        move_plan[k] = int(capped)
    
    # Update position tracking
    if "rails" in move_plan or "main" in move_plan:
        update_position(delta_x=dx_cm, delta_y=dy_cm)
    
    return move_plan

def convert_depth_to_arm_steps(depth_m, target_depth_m=0.40):
    """Convert depth to arm motor steps, respecting boundaries."""
    depth_cm = depth_m * 100
    target_cm = target_depth_m * 100
    
    error_cm = target_cm - depth_cm
    error_cm = clamp(error_cm, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE)
    
    # Respect boundary limits for arm movement
    _, _, error_cm = clamp_movement_to_limits(0.0, 0.0, error_cm)
    
    steps = int(error_cm * STEPS_PER_CM_ARM)
    
    # Update position tracking
    if steps != 0:
        delta_z = error_cm
        update_position(delta_z=delta_z)
    
    return steps

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
                mapLx, mapLy, mapRx, mapRy, Q, K_L_use = setup_stereo_rectification(w_rect, h_rect, SCALE_FOR_MATCHING)
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
    global demo_stop_flag, current_frame_left, current_frame_right, demo_mode
    
    # Reset position to origin at demo start
    reset_position()
    print("[DEMO] Position reset to origin (0, 0, 0) - rear-right")
    print_position()
    
    # Initialize arm: release and move up 50cm
    print("[DEMO] Initializing arm: releasing and moving up 50cm...")
    arm_release_steps = int(50 * STEPS_PER_CM_ARM)  # Move up 50cm
    send_motor_command(client_socket, {'arm': arm_release_steps, '_hold_motors': ['arm']})
    time.sleep(1.0)  # Wait for arm to reach position
    
    frame_count = 0
    flower_found = False  # Track if we've located a flower
    consecutive_no_detect = 0  # Track consecutive frames without detection
    
    # Square search pattern waypoints: (Y, X) in cm
    # Start (0,0) → (18,45) → (0,45) → (18,0) → back to (0,0)
    search_waypoints = [
        (18, 45),   # Front-left corner
        (0, 45),    # Front-right corner
        (18, 0),    # Rear-left corner
        (0, 0),     # Rear-right corner (start)
    ]
    current_waypoint_idx = 0
    target_waypoint = search_waypoints[current_waypoint_idx]
    
    saved_position = None  # Position before moving toward flower
    
    print("\n[DEMO] Starting automatic flower tracking demo...")
    print("[DEMO] Square search pattern: (0,0) → (18,45) → (0,45) → (18,0) → (0,0)")
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
            
            disparity = compute_stereo_disparity(rectL, rectR, K_L_use)
            
            if disparity is None:
                print(f"[DEMO] Frame {frame_count}: Disparity failed, skipping")
                send_motor_command(client_socket, {})
                time.sleep(0.05)
                continue
            
            depth_map = compute_depth_map(disparity, Q)
            
            detections = detect_flowers(frame_left, conf_threshold=YOLO_CONF)
            
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
                    pos = get_current_position()
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
                    move_step_cm = 0.3
                    if abs(dx_to_waypoint) > move_step_cm or abs(dy_to_waypoint) > move_step_cm:
                        # Normalize direction
                        dist = np.sqrt(dx_to_waypoint**2 + dy_to_waypoint**2)
                        dx_norm = (dx_to_waypoint / dist) * move_step_cm
                        dy_norm = (dy_to_waypoint / dist) * move_step_cm
                        
                        move_plan = convert_offsets_to_motor_steps(dx_norm * PIXELS_PER_CM, dy_norm * PIXELS_PER_CM)
                        motor_command = move_plan if move_plan else {}
                else:
                    print(f"[DEMO] Frame {frame_count}: Lost flower (no detect {consecutive_no_detect}/5), moving back to waypoint...")
                    # Move back toward last saved waypoint position
                    pos = get_current_position()
                    if saved_position:
                        dx_to_saved = saved_position["x"] - pos["x"]
                        dy_to_saved = saved_position["y"] - pos["y"]
                        move_plan = convert_offsets_to_motor_steps(dx_to_saved * PIXELS_PER_CM, dy_to_saved * PIXELS_PER_CM)
                        motor_command = move_plan if move_plan else {}
                    else:
                        motor_command = {}
                    
                    # Send the movement command for waypoint search
                    if motor_command:
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
                dx_cm_raw = clamp(dx / PIXELS_PER_CM, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE)
                dy_cm_raw = clamp(dy / PIXELS_PER_CM, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE)
                dx_cm_allowed, dy_cm_allowed, _ = clamp_movement_to_limits(dx_cm_raw, dy_cm_raw, 0.0)
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
                    # Debug mode: stop and display info
                    pos = get_current_position()
                    print(f"\n[DEMO] FLOWER DETECTED - DEBUG MODE")
                    print(f"[DEMO] Frame {frame_count}: Confidence={target_det['confidence']:.2f}, Depth={depth_m:.3f}m")
                    print(f"[DEMO] Current Position: X={pos['x']:.2f}cm, Y={pos['y']:.2f}cm, Z={pos['z']:.2f}cm")
                    print(f"[DEMO] Flower Offset from Center: dx={dx:.0f}px, dy={dy:.0f}px")
                    print(f"[DEMO] Movement Needed: dx={dx_cm_allowed:.2f}cm, dy={dy_cm_allowed:.2f}cm")
                    print(f"[DEMO] Motor Steps: rails={rails_steps_log}, main={main_steps_log}")
                    print(f"[DEMO] *** STOPPED FOR DEBUGGING - Press 'q' to exit ***\n")
                    time.sleep(0.05)
                    continue
                else:
                    # Normal mode: approach flower
                    # Save current position before moving toward flower
                    if saved_position is None:
                        saved_position = get_current_position().copy()
                        print(f"[DEMO] Position saved: ({saved_position['y']:.2f}, {saved_position['x']:.2f})")
                    
                    print(f"[DEMO] Frame {frame_count}: Flower found! dx={dx:.0f}px, dy={dy:.0f}px, depth={depth_m:.3f}m (conf={target_det['confidence']:.2f})")
                    print(f"[DEMO] Movement: dx={dx_cm_allowed:.2f}cm, dy={dy_cm_allowed:.2f}cm | rails={rails_steps_log} steps, main={main_steps_log} steps")

                    # Execute a small movement toward the flower center
                    move_plan = convert_offsets_to_motor_steps(dx, dy)
                    if move_plan:
                        print(f"[DEMO] Executing move toward flower: {move_plan}")
                        send_motor_command(client_socket, move_plan)
                        time.sleep(0.3)  # Short wait for movement to complete
                    else:
                        # If no movement needed (already centered), proceed to arm down
                        print(f"[DEMO] Already centered on flower, proceeding with arm down")
                    
                    # Always move arm down 30cm when flower detected
                    arm_steps = int(-30 * STEPS_PER_CM_ARM)  # Move down 30cm
                    arm_command = {'arm': arm_steps, '_hold_motors': ['arm']}  # Keep arm powered to hold position
                    print(f"[DEMO] Arm movement: {arm_steps} steps (moving down 30cm to probe flower)")
                    
                    send_motor_command(client_socket, arm_command)
                    time.sleep(0.5)  # Wait for arm to reach flower
                    
                    # Pollinate the flower
                    pollinate(client_socket, vibrate_duration_ms=500, van_de_graaf_duration_ms=500, repeat=1)
                    
                    # Move arm back up 30cm
                    arm_up_steps = int(30 * STEPS_PER_CM_ARM)  # Move up 30cm
                    arm_command_up = {'arm': arm_up_steps, '_hold_motors': ['arm']}  # Keep arm held after retract
                    send_motor_command(client_socket, arm_command_up)
                    print(f"[DEMO] Arm retracted: {arm_up_steps} steps (moving up 30cm)")
                    time.sleep(0.5)  # Wait for arm to retract
                    
                    # Move back to saved position
                    if saved_position:
                        pos = get_current_position()
                        dx_to_saved = saved_position["x"] - pos["x"]
                        dy_to_saved = saved_position["y"] - pos["y"]
                        print(f"[DEMO] Moving back to saved position: ({saved_position['y']:.2f}, {saved_position['x']:.2f})")
                        move_plan = convert_offsets_to_motor_steps(dx_to_saved * PIXELS_PER_CM, dy_to_saved * PIXELS_PER_CM)
                        if move_plan:
                            send_motor_command(client_socket, move_plan)
                            time.sleep(0.5)
                        saved_position = None  # Clear saved position for next flower
                    
                    # Don't send empty command at the end of frame when flower detected
                    time.sleep(0.05)
                    continue
            
            # Display
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
# Command Interface
# =============================
def print_help():
    """Print available commands."""
    print("\n" + "="*60)
    print("LAPTOP CLIENT - MANUAL CONTROL MODE")
    print("="*60)
    print("\nCoordinate System:")
    print("  Origin (0,0,0) = rear-right corner")
    print("  X (rails): 0-45 cm (0=right, 45=left)")
    print("  Y (main):  0-18 cm (0=rear, 18=front)")
    print("  Z (arm):   0-20 cm (0=down/released, 20=up)")
    print("\nAvailable Commands:")
    print("  status                   - Show current position and limits")
    print("  reset                    - Return to origin (0,0,0) and send reset command")
    print("  move <motor> <steps>     - Move specific motor")
    print("                             Motors: rails, main, arm")
    print("                             Steps: positive or negative integer")
    print("  movecm <motor> <cm>      - Move motor by centimeters (rails/main/arm)")
    print("                             Uses calibration to convert cm -> steps")
    print("                             Enforces boundary limits")
    print("  arm <steps>              - Shortcut for 'move arm <steps>' (held)")
    print("  release                  - Release all motors (stop holding position)")
    print("  demo                     - Start automatic flower detection demo")
    print("                             (resets position to origin)")
    print("  stop                     - Stop demo mode")
    print("  vibrate <ms>             - Run vibration motor for N milliseconds")
    print("  vdg <ms>        - Run van de Graaf motor for N milliseconds")
    print("  pollinate [v_ms] [vdg_ms] [repeat]")
    print("                           - Run pollination sequence")
    print("                             Default: pollinate 500 500 1")
    print("  help                     - Show this help message")
    print("  quit                     - Exit program")
    print("\nExamples:")
    print("  status                   - Show current position")
    print("  reset                    - Go back to origin")
    print("  move rails 100           - Move rails motor 100 steps")
    print("  vibrate 500              - Vibrate for 500ms")
    print("  vdg 300                  - Van de Graaf for 300ms")
    print("  pollinate 500 500 2      - Pollinate 2 cycles (500ms vibrate + 500ms vdg each)")
    print("="*60 + "\n")

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
            print_position()
        
        elif command == "reset":
            print("[COMMAND] Resetting all motors to home position...")
            motor_command = {"rails": 0, "main": 0, "arm": 0}
            send_motor_command(client_socket, motor_command)
            reset_position()
            arm_is_at_bottom = True
            print("[COMMAND] Reset command sent! Position reset to origin (0, 0, 0)")
            print_position()
        
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
                dist_cm, _, _ = clamp_movement_to_limits(dist_cm, 0.0, 0.0)
            elif motor_name == "main":
                _, dist_cm, _ = clamp_movement_to_limits(0.0, dist_cm, 0.0)
            elif motor_name == "arm":
                if arm_is_at_bottom and dist_cm > 0:
                    adjusted_dist_cm = dist_cm + ARM_BOTTOM_OFFSET_CM
                    print(f"[COMMAND] Arm at bottom, adding {ARM_BOTTOM_OFFSET_CM} cm offset")
                    print(f"[COMMAND] Adjusted distance: {dist_cm:.2f} cm -> {adjusted_dist_cm:.2f} cm")
                    dist_cm = adjusted_dist_cm
                    arm_is_at_bottom = False
                elif dist_cm < 0:
                    arm_is_at_bottom = False
                
                _, _, dist_cm = clamp_movement_to_limits(0.0, 0.0, dist_cm)

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
                update_position(delta_x=dist_cm)
            elif motor_name == "main":
                update_position(delta_y=dist_cm)
            elif motor_name == "arm":
                update_position(delta_z=dist_cm)
            
            print_position()
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
            update_position(delta_z=delta_z)
            print_position()
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
            print_position()
        
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
    reset_position()
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
