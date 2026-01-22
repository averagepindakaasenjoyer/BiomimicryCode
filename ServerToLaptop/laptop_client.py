"""
Laptop Client for Stereo Vision Processing

Receives stereo frames from Pi, performs:
- YOLO flower detection
- Stereo depth estimation
- Motor movement calculation

Sends motor commands back to Pi for execution.
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

# =============================
# Network Configuration
# =============================
PI_IP = "145.109.40.215"  # Replace with your Pi's IP
PORT = 8000

# =============================
# Vision Configuration
# =============================
# Paths (relative to parent directory)
YOLO_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../current_best_yolo.pt"))
CALIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Cam_Params/stereo_charuco_calibration_16cm_rico.npz'))

# Detection parameters
CONFIDENCE_THRESHOLD = 0.6
YOLO_CONF = 0.5
SCALE_FOR_MATCHING = 0.5

# Depth estimation
MIN_DEPTH = 0.25  # Original: 0.25
MAX_DEPTH = 2.0   # Original: 2.0
EXPECTED_DISTANCE = 0.40  # Original: 0.40 (target working distance in meters)

# Display
SHOW_DEBUG = True
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# =============================
# Motor Parameters (from FlowerTrackingStereo.py)
# =============================
WHEEL_DIAMETER_CM = 2.5
CIRCUMFERENCE_CM = WHEEL_DIAMETER_CM * np.pi
STEPS_PER_REV = 200
STEPS_PER_CM = STEPS_PER_REV / CIRCUMFERENCE_CM
scale_move = 0.1

PIXELS_PER_CM = 10.0
MAX_CM_PER_CYCLE = 10.0
MAX_STEPS_PER_CYCLE = int(MAX_CM_PER_CYCLE * STEPS_PER_CM)

# Arm parameters
WHEEL_DIAMETER_CM_ARM = 4.3
CIRCUMFERENCE_CM_ARM = WHEEL_DIAMETER_CM_ARM * np.pi
STEPS_PER_REV_ARM = 200
STEPS_PER_CM_ARM = STEPS_PER_REV_ARM / CIRCUMFERENCE_CM_ARM

# Motor direction mapping (from FlowerTrackingStereo.py)
direction_dict = {
    "front": [("main", 1), ("main", 1)],
    "rear": [("main", -1), ("main", -1)],
    "right": [("rails", 1), ("rails", 1)],
    "left": [("rails", -1), ("rails", -1)],
    "up": [("arm", 1)],
    "down": [("arm", -1)],
}

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
# Utility Functions
# =============================
def clamp(n, small, large):
    return max(small, min(n, large))

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
    
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_L_use, D_L, K_R_use, D_R, (w, h), R, T, alpha=1
    )
    
    mapLx, mapLy = cv2.initUndistortRectifyMap(
        K_L_use, D_L, R1, P1, (w, h), cv2.CV_32FC1
    )
    mapRx, mapRy = cv2.initUndistortRectifyMap(
        K_R_use, D_R, R2, P2, (w, h), cv2.CV_32FC1
    )
    
    return mapLx, mapLy, mapRx, mapRy, Q, K_L_use

def compute_stereo_disparity(rectL, rectR, K_L_use):
    """Compute disparity map from rectified stereo pair."""
    # Validate input shapes
    if rectL.shape != rectR.shape:
        print(f"[WARNING] Shape mismatch: rectL={rectL.shape}, rectR={rectR.shape}")
        return None
    
    # Convert to grayscale
    if rectL.ndim == 3:
        rectL_gray = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        rectR_gray = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    else:
        rectL_gray = rectL
        rectR_gray = rectR
    
    # Validate grayscale conversion
    if rectL_gray.dtype != np.uint8:
        rectL_gray = rectL_gray.astype(np.uint8)
    if rectR_gray.dtype != np.uint8:
        rectR_gray = rectR_gray.astype(np.uint8)
    
    # Apply CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    rectL_proc = clahe.apply(rectL_gray)
    rectR_proc = clahe.apply(rectR_gray)
    
    # Calculate numDisparities with validation
    try:
        expected_disp = (K_L_use[0, 0] * baseline) / EXPECTED_DISTANCE
        num_disp = int(np.ceil(expected_disp * 1.8 / 16.0) * 16)
        # Clamp to safe range (must be multiple of 16)
        # ORIGINAL RANGE: max(160, min(num_disp, 640)) - caused OutOfMemory error
        # FIXED RANGE: max(16, min(num_disp, 256)) - safe for real-time processing
        num_disp = max(16, min(num_disp, 256))
        # Ensure it's a multiple of 16
        num_disp = (num_disp // 16) * 16
        if num_disp < 16:
            num_disp = 16
    except Exception as e:
        print(f"[WARNING] Disparity calculation error: {e}, using default 128")
        num_disp = 128
    
    # Adjust block size based on image resolution
    h, w = rectL_proc.shape
    # ORIGINAL: always block_size = 5
    # FIXED: adaptive - 5 for larger images, 3 for smaller resolution (better for 320x240)
    block_size = 5 if min(h, w) > 200 else 3
    
    print(f"[DEBUG] Disparity params: numDisp={num_disp}, blockSize={block_size}, imgSize={w}x{h}")
    
    stereo_matcher = cv2.StereoSGBM.create(
        minDisparity=0,  # ORIGINAL: 0
        numDisparities=num_disp,  # ORIGINAL: max(160, min(num_disp, 640))
        blockSize=block_size,  # ORIGINAL: always 5 (now adaptive: 3 or 5)
        P1=8 * block_size**2,  # ORIGINAL: 8 * 5**2 = 200 (now scales with block_size)
        P2=32 * block_size**2,  # ORIGINAL: 32 * 5**2 = 800 (now scales with block_size)
        disp12MaxDiff=1,  # ORIGINAL: 1
        uniquenessRatio=10,  # ORIGINAL: 6 (increased for robustness)
        speckleWindowSize=100,  # ORIGINAL: 80 (increased slightly)
        speckleRange=32,  # ORIGINAL: 32
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # ORIGINAL: same
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
    """Select one flower to track (prioritize closest, then most centered)."""
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
        
        # Distance from frame center
        dist_from_center = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
        
        # Prioritize by depth (closer is better) then by centering
        depth = depth_stats['median']
        score = depth * 0.7 + dist_from_center * 0.0003
        
        if score < best_score:
            best_score = score
            best_idx = i
    
    if best_idx is not None:
        return detections[best_idx], depth_stats_list[best_idx]
    
    return None, None

def convert_offsets_to_motor_steps(dx_pixels, dy_pixels):
    """Convert pixel offsets to motor steps."""
    dx_cm = dx_pixels / PIXELS_PER_CM
    dy_cm = dy_pixels / PIXELS_PER_CM
    
    dx_cm = clamp(dx_cm, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE)
    dy_cm = clamp(dy_cm, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE)
    
    move_plan = {}
    
    # X axis movement (left/right rails)
    if abs(dx_cm) >= 0.5:
        if dx_cm > 0:
            entries = direction_dict["left"]
        else:
            entries = direction_dict["right"]
        steps_for_cm = abs(dx_cm) * STEPS_PER_CM * scale_move
        for motor_name, multiplier in entries:
            motor_steps = int(multiplier * steps_for_cm)
            move_plan[motor_name] = move_plan.get(motor_name, 0) + motor_steps
    
    # Y axis movement (main rails forward/backward)
    if abs(dy_cm) >= 0.5:
        if dy_cm > 0:
            entries = direction_dict["rear"]
        else:
            entries = direction_dict["front"]
        steps_for_cm = abs(dy_cm) * STEPS_PER_CM * scale_move
        for motor_name, multiplier in entries:
            motor_steps = int(multiplier * steps_for_cm)
            move_plan[motor_name] = move_plan.get(motor_name, 0) + motor_steps
    
    # Clamp to max steps
    for k in list(move_plan.keys()):
        capped = clamp(move_plan[k], -MAX_STEPS_PER_CYCLE, MAX_STEPS_PER_CYCLE)
        move_plan[k] = int(capped)
    
    return move_plan

def convert_depth_to_arm_steps(depth_m, target_depth_m=0.40):
    """Convert depth to arm motor steps to reach target depth."""
    depth_cm = depth_m * 100
    target_cm = target_depth_m * 100
    
    error_cm = target_cm - depth_cm
    error_cm = clamp(error_cm, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE)
    steps = int(error_cm * STEPS_PER_CM_ARM)
    return steps

# =============================
# Network Communication
# =============================
def receive_stereo_frames(client_socket, data, payload_size):
    """Receive stereo frame pair from Pi."""
    # Receive message size
    while len(data) < payload_size:
        packet = client_socket.recv(4096)
        if not packet:
            return None, None, data
        data += packet
    
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]
    
    # Receive frame data
    while len(data) < msg_size:
        packet = client_socket.recv(4096)
        if not packet:
            return None, None, data
        data += packet
    
    frame_data = data[:msg_size]
    data = data[msg_size:]
    
    # Deserialize
    frames_dict = pickle.loads(frame_data)
    
    # Decode JPEG images
    frame_left = cv2.imdecode(frames_dict['left'], cv2.IMREAD_COLOR)
    frame_right = cv2.imdecode(frames_dict['right'], cv2.IMREAD_COLOR)
    
    return frame_left, frame_right, data

def send_motor_command(client_socket, command_dict):
    """Send motor command dictionary to Pi."""
    try:
        data = pickle.dumps(command_dict)
        client_socket.send(data)
    except Exception as e:
        print(f"[Laptop] Error sending command: {e}")

# =============================
# Main Processing Loop
# =============================
def main():
    # Connect to Pi
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((PI_IP, PORT))
    print(f"[Laptop] Connected to Pi at {PI_IP}:{PORT}")
    
    data = b""
    payload_size = struct.calcsize("Q")
    
    # Initialize stereo rectification (will be set once we get first frame)
    mapLx, mapLy, mapRx, mapRy, Q, K_L_use = None, None, None, None, None, None
    stereo_initialized = False
    
    frame_count = 0
    
    try:
        while True:
            # Receive stereo frames
            frame_left, frame_right, data = receive_stereo_frames(client_socket, data, payload_size)
            
            if frame_left is None or frame_right is None:
                print("[Laptop] Failed to receive frames")
                break
            
            frame_count += 1
            
            # Initialize stereo rectification on first frame
            if not stereo_initialized:
                h, w = frame_left.shape[:2]
                w_rect = int(w * SCALE_FOR_MATCHING)
                h_rect = int(h * SCALE_FOR_MATCHING)
                mapLx, mapLy, mapRx, mapRy, Q, K_L_use = setup_stereo_rectification(w_rect, h_rect, SCALE_FOR_MATCHING)
                stereo_initialized = True
                print(f"[Laptop] Stereo initialized for {w}x{h} -> {w_rect}x{h_rect}")
            
            # Downscale for stereo matching
            h, w = frame_left.shape[:2]
            w_rect = int(w * SCALE_FOR_MATCHING)
            h_rect = int(h * SCALE_FOR_MATCHING)
            frame_l_scaled = cv2.resize(frame_left, (w_rect, h_rect))
            frame_r_scaled = cv2.resize(frame_right, (w_rect, h_rect))
            
            # Rectify
            rectL = cv2.remap(frame_l_scaled, mapLx, mapLy, cv2.INTER_LINEAR)
            rectR = cv2.remap(frame_r_scaled, mapRx, mapRy, cv2.INTER_LINEAR)
            
            # Compute depth
            disparity = compute_stereo_disparity(rectL, rectR, K_L_use)
            
            if disparity is None:
                print(f"[Laptop] Frame {frame_count}: Disparity computation failed, skipping")
                send_motor_command(client_socket, {})
                continue
            
            depth_map = compute_depth_map(disparity, Q)
            
            # Detect flowers on full-res left frame
            detections = detect_flowers(frame_left, conf_threshold=YOLO_CONF)
            
            # Estimate depth for each detection
            depth_stats_list = []
            for det in detections:
                scaled_box = tuple(int(coord * SCALE_FOR_MATCHING) for coord in det['box'])
                depth_stats = estimate_roi_depth(depth_map, scaled_box)
                depth_stats_list.append(depth_stats)
            
            # Select target flower
            target_det, target_depth = select_target_flower(detections, depth_stats_list, w, h)
            
            # Calculate motor command
            motor_command = {}
            
            if target_det is None:
                print(f"[Laptop] Frame {frame_count}: No valid flower detected, searching...")
                # Slow backward movement to search
                motor_command = {"main": int(0.5 * STEPS_PER_CM)}
            else:
                # Calculate offsets from center
                x1, y1, x2, y2 = target_det['box']
                flower_center_x = (x1 + x2) / 2
                flower_center_y = (y1 + y2) / 2
                
                dx = flower_center_x - (w / 2)
                dy = flower_center_y - (h / 2)
                
                depth_m = target_depth['median']
                
                print(f"[Laptop] Frame {frame_count}: Target at dx={dx:.0f}px, dy={dy:.0f}px, depth={depth_m:.3f}m")
                
                # Plan movement
                motor_command = convert_offsets_to_motor_steps(dx, dy)
                
                # Add arm adjustment if well-centered
                if abs(dx) < 30 and abs(dy) < 30:
                    arm_steps = convert_depth_to_arm_steps(depth_m, target_depth_m=0.40)
                    if abs(arm_steps) > 10:
                        motor_command['arm'] = arm_steps
                        print(f"[Laptop] Adding arm adjustment: {arm_steps} steps")
            
            # Send motor command to Pi
            send_motor_command(client_socket, motor_command)
            
            # Display debug window
            if SHOW_DEBUG:
                display_frame = frame_left.copy()
                
                # Draw all detections
                for det, depth_stats in zip(detections, depth_stats_list):
                    x1, y1, x2, y2 = det['box']
                    is_target = (det == target_det)
                    color = (0, 255, 0) if is_target else (128, 128, 128)
                    thickness = 3 if is_target else 1
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    if depth_stats:
                        label = f"{det['confidence']:.2f} | {depth_stats['median']:.2f}m"
                        cv2.putText(display_frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw center crosshair
                cv2.line(display_frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 0, 0), 2)
                cv2.line(display_frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 0, 0), 2)
                
                # Draw offset to target
                if target_det:
                    x1, y1, x2, y2 = target_det['box']
                    target_x = int((x1 + x2) / 2)
                    target_y = int((y1 + y2) / 2)
                    cv2.line(display_frame, (w//2, h//2), (target_x, target_y), (0, 255, 255), 2)
                    cv2.circle(display_frame, (target_x, target_y), 10, (0, 255, 0), -1)
                
                # Info overlay
                cv2.putText(display_frame, f"Frame: {frame_count} | Detections: {len(detections)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if motor_command:
                    cmd_str = ", ".join([f"{k}:{v}" for k, v in motor_command.items()])
                    cv2.putText(display_frame, f"Command: {cmd_str}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                display_frame = cv2.resize(display_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                cv2.imshow("Laptop - Flower Tracking", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
    
    except KeyboardInterrupt:
        print("[Laptop] Interrupted by user")
    except Exception as e:
        print(f"[Laptop] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print("[Laptop] Cleanup complete")

if __name__ == "__main__":
    main()
