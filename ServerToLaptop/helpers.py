"""
Helper utilities for position tracking, movement calculations, and math operations.
"""

import numpy as np

# =============================
# Position and Movement Helpers
# =============================

def clamp(n, small, large):
    """Clamp value between bounds."""
    return max(small, min(n, large))


def get_current_position(position_lock, current_position):
    """Get current position safely (thread-safe)."""
    with position_lock:
        return current_position.copy()


def update_position(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX, 
                   LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX,
                   delta_x=0.0, delta_y=0.0, delta_z=0.0):
    """Update position and enforce limits."""
    with position_lock:
        new_x = current_position["x"] + delta_x
        new_y = current_position["y"] + delta_y
        new_z = current_position["z"] + delta_z
        
        # Clamp to limits
        current_position["x"] = clamp(new_x, LIMIT_X_MIN, LIMIT_X_MAX)
        current_position["y"] = clamp(new_y, LIMIT_Y_MIN, LIMIT_Y_MAX)
        current_position["z"] = clamp(new_z, LIMIT_Z_MIN, LIMIT_Z_MAX)


def reset_position(position_lock, current_position):
    """Reset to origin (0,0,0) - rear-right."""
    with position_lock:
        current_position["x"] = 0.0
        current_position["y"] = 0.0
        current_position["z"] = 0.0


def print_position(position_lock, current_position):
    """Print current position and limits."""
    pos = get_current_position(position_lock, current_position)
    print("\n" + "="*60)
    print("CURRENT POSITION")
    print("="*60)
    print(f"  X (rails):  {pos['x']:6.2f} cm  [0.00 - 45.00] (0=right, 45=left)")
    print(f"  Y (main):   {pos['y']:6.2f} cm  [0.00 - 18.00] (0=rear, 18=front)")
    print(f"  Z (arm):    {pos['z']:6.2f} cm  [0.00 - 20.00] (0=down, 20=up)")
    print(f"\nOrigin (0,0,0) is rear-right corner")
    print("="*60 + "\n")


def clamp_movement_to_limits(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX,
                             LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX,
                             delta_x_cm, delta_y_cm, delta_z_cm):
    """Clamp desired movement to stay within limits."""
    pos = get_current_position(position_lock, current_position)
    
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


# =============================
# Flower Tracking Helpers
# =============================

def is_flower_already_pollinated(pollinated_flowers, x, y, exclusion_radius_cm):
    """Check if a flower at (x, y) world coordinates is within the exclusion radius of any pollinated flower.
    
    Args:
        pollinated_flowers: List of pollinated flower positions
        x: X coordinate in world space (cm)
        y: Y coordinate in world space (cm)
        exclusion_radius_cm: Radius around pollinated flowers to exclude
        
    Returns:
        True if flower is too close to a pollinated flower, False otherwise
    """
    for f in pollinated_flowers:
        dist = np.hypot(f["x"] - x, f["y"] - y)
        if dist < exclusion_radius_cm:
            print(f"[POLLINATION] Flower at ({x:.2f}, {y:.2f}) is within {dist:.2f}cm of pollinated flower at ({f['x']:.2f}, {f['y']:.2f}) - EXCLUDING")
            return True
    return False


def mark_flower_as_pollinated(pollinated_flowers, x, y):
    """Mark a flower as pollinated at world coordinates (x, y).
    
    Args:
        pollinated_flowers: List to append pollinated flower data to
        x: X coordinate in world space (cm)
        y: Y coordinate in world space (cm)
    """
    import time
    pollinated_flowers.append({
        "x": x,
        "y": y,
        "timestamp": time.time()
    })
    print(f"[POLLINATION] ✓ Flower marked as pollinated at world position ({x:.2f}, {y:.2f})")
    print(f"[POLLINATION] Total pollinated flowers: {len(pollinated_flowers)}")
    if len(pollinated_flowers) > 1:
        print(f"[POLLINATION] Pollinated flower locations:")
        for i, f in enumerate(pollinated_flowers):
            print(f"  {i+1}. ({f['x']:.2f}, {f['y']:.2f})")


# Legacy compatibility functions
def is_flower_already_visited(visited_flowers, x, y, FLOWER_VISIT_RADIUS_CM):
    """Legacy function - use is_flower_already_pollinated instead."""
    return is_flower_already_pollinated(visited_flowers, x, y, FLOWER_VISIT_RADIUS_CM)


def mark_flower_as_visited(visited_flowers, x, y):
    """Legacy function - use mark_flower_as_pollinated instead."""
    mark_flower_as_pollinated(visited_flowers, x, y)


# =============================
# Image Processing Helpers
# =============================

def rotate_frame(frame, rotation_degrees):
    """Rotate frame by specified degrees (0, 90, 180, 270)."""
    import cv2
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
    """Scale camera intrinsic matrix."""
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale
    K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale
    K_scaled[1, 2] *= scale
    return K_scaled


# =============================
# Stereo Vision Helpers
# =============================

def setup_stereo_rectification(K_L, D_L, K_R, D_R, R, T, w, h, scale):
    """Setup stereo rectification maps."""
    import cv2
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


def compute_stereo_disparity(rectL, rectR, K_L_use, baseline, EXPECTED_DISTANCE):
    """Compute disparity map from rectified stereo pair."""
    import cv2
    
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
    import cv2
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    depth_map = points_3d[:, :, 2]
    
    invalid_mask = (disparity <= 0) | ~np.isfinite(depth_map)
    depth_map[invalid_mask] = 0.0
    
    return depth_map


# =============================
# YOLO Detection Helpers
# =============================

def detect_flowers(frame, model, device, conf_threshold, CONFIDENCE_THRESHOLD):
    """Run YOLO detection on frame."""
    import cv2
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


def find_flower_center_in_bbox(frame, bbox, use_advanced=False):
    """Find flower center within bounding box.
    
    Args:
        frame: Image frame
        bbox: Bounding box as (x1, y1, x2, y2)
        use_advanced: If True, find yellow region center; if False, return bbox center
        
    Returns:
        Tuple of (center_x, center_y) in frame coordinates
    """
    import cv2
    
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Simple center estimation
    if not use_advanced:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y
    
    # Advanced: Find yellow region within bbox
    try:
        # Crop to bounding box
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            # Fallback to center if crop is empty
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            return center_x, center_y
        
        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Define yellow color range in HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        # Create mask for yellow color
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Apply morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (yellow region)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get center of largest contour
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                yellow_center_x = int(M['m10'] / M['m00'])
                yellow_center_y = int(M['m01'] / M['m00'])
                
                # Convert back to full frame coordinates
                center_x = x1 + yellow_center_x
                center_y = y1 + yellow_center_y
                print(f"[FLOWER] Advanced estimation: yellow region center at ({center_x}, {center_y})")
                return center_x, center_y
        
        # No yellow region found, fallback to center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        print(f"[FLOWER] No yellow region found in bbox, using center ({center_x}, {center_y})")
        return center_x, center_y
        
    except Exception as e:
        print(f"[FLOWER] Error in advanced estimation: {e}")
        # Fallback to simple center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y


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


# =============================
# Motor Movement Calculation
# =============================

def convert_offsets_to_motor_steps(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX,
                                   LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX,
                                   dx_pixels, dy_pixels, PIXELS_PER_CM, MAX_CM_PER_CYCLE,
                                   MAX_STEPS_PER_CYCLE, STEPS_PER_CM, scale_move, 
                                   RAILS_CALIBRATION, MAIN_CALIBRATION, direction_dict):
    """Convert pixel offsets to motor steps, respecting boundaries."""
    dx_cm = dx_pixels / PIXELS_PER_CM
    dy_cm = dy_pixels / PIXELS_PER_CM
    
    dx_cm = clamp(dx_cm, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE)
    dy_cm = clamp(dy_cm, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE)
    
    # Respect boundary limits
    dx_cm, dy_cm, _ = clamp_movement_to_limits(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX,
                                               LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX,
                                               dx_cm, dy_cm, 0.0)
    
    move_plan = {}
    
    # Lower threshold to 0.1cm for smaller movements
    if abs(dx_cm) >= 0.1:
        if dx_cm > 0:
            entries = direction_dict["left"]
        else:
            entries = direction_dict["right"]
        steps_for_cm = abs(dx_cm) * STEPS_PER_CM * scale_move * RAILS_CALIBRATION
        for motor_name, multiplier in entries:
            motor_steps = int(multiplier * steps_for_cm)
            move_plan[motor_name] = move_plan.get(motor_name, 0) + motor_steps
    
    if abs(dy_cm) >= 0.1:
        if dy_cm > 0:
            entries = direction_dict["rear"]
        else:
            entries = direction_dict["front"]
        steps_for_cm = abs(dy_cm) * STEPS_PER_CM * scale_move * MAIN_CALIBRATION
        for motor_name, multiplier in entries:
            motor_steps = int(multiplier * steps_for_cm)
            move_plan[motor_name] = move_plan.get(motor_name, 0) + motor_steps
    
    for k in list(move_plan.keys()):
        capped = clamp(move_plan[k], -MAX_STEPS_PER_CYCLE, MAX_STEPS_PER_CYCLE)
        move_plan[k] = int(capped)
    
    # Update position tracking
    if "rails" in move_plan or "main" in move_plan:
        update_position(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX,
                       LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX,
                       delta_x=dx_cm, delta_y=dy_cm)
    
    return move_plan


def convert_depth_to_arm_steps(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX,
                               LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX,
                               depth_m, MAX_CM_PER_CYCLE, STEPS_PER_CM_ARM, target_depth_m=0.40):
    """Convert depth to arm motor steps, respecting boundaries."""
    depth_cm = depth_m * 100
    target_cm = target_depth_m * 100
    
    error_cm = target_cm - depth_cm
    error_cm = clamp(error_cm, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE)
    
    # Respect boundary limits for arm movement
    _, _, error_cm = clamp_movement_to_limits(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX,
                                              LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX,
                                              0.0, 0.0, error_cm)
    
    steps = int(error_cm * STEPS_PER_CM_ARM)
    
    # Update position tracking
    if steps != 0:
        delta_z = error_cm
        update_position(position_lock, current_position, LIMIT_X_MIN, LIMIT_X_MAX,
                       LIMIT_Y_MIN, LIMIT_Y_MAX, LIMIT_Z_MIN, LIMIT_Z_MAX,
                       delta_z=delta_z)
    
    return steps


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
    print("  keys / keyboard          - Enter real-time keyboard control mode")
    print("                             W=Forward, S=Backward, A=Left, D=Right")
    print("                             O=ArmUp, P=ArmDown, Q/ESC=Exit")
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
    print("  keys                     - Enter keyboard control mode")
    print("  vibrate 500              - Vibrate for 500ms")
    print("  vdg 300                  - Van de Graaf for 300ms")
    print("  pollinate 500 500 2      - Pollinate 2 cycles (500ms vibrate + 500ms vdg each)")
    print("="*60 + "\n")
