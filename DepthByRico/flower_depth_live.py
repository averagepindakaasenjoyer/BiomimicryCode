"""
Live flower depth estimation from stereo cameras using YOLO detection.

Captures stereo video from two USB cameras, runs YOLO detection for flowers,
and estimates depth for detected bounding boxes (confidence > 0.8).

Controls:
  - ESC: Exit
  - SPACE: Pause/resume
  - 'f': Toggle fullscreen
  - 'r': Reset camera captures (re-initialize stereo pipelines)
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
from dotenv import load_dotenv
import time

load_dotenv()

# =============================
# Configuration
# =============================
YOLO_MODEL_PATH = "current_best_yolo.pt"
CALIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Cam_Params/stereo_charuco_calibration_16cm_rico.npz'))

CAM_LEFT = 1
CAM_RIGHT = 2

# Stereo parameters
CONFIDENCE_THRESHOLD = 0.6
YOLO_CONF = 0.5  # internal YOLO threshold (relaxed; filter by CONFIDENCE_THRESHOLD after)

# Frame processing
TARGET_FPS = 10  # adjustable frame rate
SCALE_FOR_MATCHING = 0.5  # downscale before stereo matching to fit disparity range

# Depth estimation
MIN_DEPTH = 0.25  # meters
MAX_DEPTH = 2.0   # meters
EXPECTED_DISTANCE = 0.40  # typical object distance for reference

# Display
DISPLAY_WIDTH = 1600
DISPLAY_HEIGHT = 900
SHOW_DISPARITY = True
SHOW_CENTER_CROSS = True

# =============================
# Load YOLO and calibration
# =============================
print(f"[INFO] Loading YOLO model from {YOLO_MODEL_PATH}...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")
model = YOLO(YOLO_MODEL_PATH)

print(f"[INFO] Loading calibration from {CALIB_PATH}...")
calib = np.load(CALIB_PATH)
K_L = calib["K_left"]
D_L = calib["D_left"]
K_R = calib["K_right"]
D_R = calib["D_right"]
R = calib["R"]
T = calib["T"]
baseline = np.linalg.norm(T)
print(f"[INFO] Baseline: {baseline:.4f} m")

# =============================
# Helper functions
# =============================
def scale_intrinsics(K, scale):
    """Scale camera intrinsics for downsampled images."""
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale
    K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale
    K_scaled[1, 2] *= scale
    return K_scaled

def setup_stereo_rectification(w, h, scale):
    """Setup stereo rectification maps and Q matrix."""
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

def rectify_frames(frame_l, frame_r, mapLx, mapLy, mapRx, mapRy):
    """Rectify stereo frame pair."""
    rectL = cv2.remap(frame_l, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(frame_r, mapRx, mapRy, cv2.INTER_LINEAR)
    return rectL, rectR

def compute_stereo_disparity(rectL, rectR, K_L_use):
    """Compute disparity and depth map."""
    # Convert to grayscale
    if rectL.ndim == 3:
        rectL_gray = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        rectR_gray = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    else:
        rectL_gray = rectL
        rectR_gray = rectR
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    rectL_proc = clahe.apply(rectL_gray)
    rectR_proc = clahe.apply(rectR_gray)
    
    # Compute expected disparity at target distance
    expected_disp = (K_L_use[0, 0] * baseline) / EXPECTED_DISTANCE
    num_disp = int(np.ceil(expected_disp * 1.8 / 16.0) * 16)
    num_disp = max(160, min(num_disp, 640))
    
    # StereoSGBM
    stereo = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=5,
        P1=8 * 5**2,
        P2=32 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=6,
        speckleWindowSize=80,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    disparity = stereo.compute(rectL_proc, rectR_proc).astype(np.float32) / 16.0
    
    return disparity, rectL, rectR

def compute_depth_map(disparity, Q):
    """Compute depth map from disparity."""
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    depth_map = points_3d[:, :, 2]
    
    # Mask invalid disparities
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
            
            # Convert to numpy
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
    """Estimate depth for a bounding box ROI."""
    x1, y1, x2, y2 = roi_box
    
    # Apply scale factor if ROI is from full-res but depth_map is scaled
    if scale_factor != 1.0:
        x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)
    
    # Bounds check
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
        'min': float(valid_depths.min()),
        'max': float(valid_depths.max()),
        'mean': float(valid_depths.mean()),
        'median': float(np.median(valid_depths)),
        'valid_pixels': len(valid_depths),
        'total_pixels': roi_depth.size,
        'coverage': 100.0 * len(valid_depths) / roi_depth.size
    }

def draw_detections_with_depth(frame, detections, depth_stats_list, thickness=2):
    """Draw bounding boxes and depth info on frame."""
    for det, depth_stats in zip(detections, depth_stats_list):
        if depth_stats is None:
            continue
        
        x1, y1, x2, y2 = det['box']
        conf = det['confidence']
        depth_med = depth_stats['median']
        coverage = depth_stats['coverage']
        
        # Color based on depth validity
        color = (0, 255, 0) if depth_stats['valid_pixels'] > 0 else (0, 0, 255)
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"Flower {conf:.2f}\nD:{depth_med:.3f}m ({coverage:.0f}%)"
        y_text = max(y1 - 6, 25)
        for i, line in enumerate(label.split('\n')):
            cv2.putText(frame, line, (x1, y_text + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    return frame

def visualize_disparity(disparity, scale=8):
    """Visualize disparity map."""
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = disp_vis.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    return disp_color

# =============================
# Main loop
# =============================
def main():
    print(f"[INFO] Opening cameras {CAM_LEFT} and {CAM_RIGHT}...")
    cap_left = cv2.VideoCapture(CAM_LEFT)
    cap_right = cv2.VideoCapture(CAM_RIGHT)
    
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("[ERROR] Could not open cameras!")
        return
    
    # Set resolution
    for cap in [cap_left, cap_right]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get resolution
    w = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Capture resolution: {w}x{h}")
    
    # Setup stereo
    print("[INFO] Setting up stereo rectification...")
    w_rect = int(w * SCALE_FOR_MATCHING)
    h_rect = int(h * SCALE_FOR_MATCHING)
    mapLx, mapLy, mapRx, mapRy, Q, K_L_use = setup_stereo_rectification(w_rect, h_rect, SCALE_FOR_MATCHING)
    
    paused = False
    fullscreen = False
    window_name = "Flower Depth Estimation"
    
    frame_count = 0
    fps_clock = time.time()
    display_fps = 0.0
    
    try:
        while True:
            # Capture frames
            ret_l, frame_l = cap_left.read()
            ret_r, frame_r = cap_right.read()
            
            if not ret_l or not ret_r:
                print("[ERROR] Failed to read frames")
                break
            
            if not paused:
                frame_count += 1
                
                # Downscale for stereo
                frame_l_scaled = cv2.resize(frame_l, (w_rect, h_rect))
                frame_r_scaled = cv2.resize(frame_r, (w_rect, h_rect))
                
                # Rectify
                try:
                    rectL, rectR = rectify_frames(frame_l_scaled, frame_r_scaled, mapLx, mapLy, mapRx, mapRy)
                except Exception as e:
                    print(f"[WARNING] Rectification failed: {e}")
                    rectL, rectR = frame_l_scaled, frame_r_scaled
                
                # Compute disparity/depth
                try:
                    disparity, _, _ = compute_stereo_disparity(rectL, rectR, K_L_use)
                    depth_map = compute_depth_map(disparity, Q)
                except Exception as e:
                    print(f"[WARNING] Stereo compute failed: {e}")
                    depth_map = np.zeros((h_rect, w_rect), dtype=np.float32)
                
                # Detect flowers on full-resolution frame
                detections = detect_flowers(frame_l, conf_threshold=YOLO_CONF)
                
                # Estimate depth for each detection (scale back to full-res, then scale down for depth_map)
                depth_stats_list = []
                for det in detections:
                    # Scale detection box to depth_map coordinates
                    scaled_box = tuple(int(coord * SCALE_FOR_MATCHING) for coord in det['box'])
                    depth_stats = estimate_roi_depth(depth_map, scaled_box)
                    depth_stats_list.append(depth_stats)
                
                # Prepare display frame
                display_frame = frame_l.copy()
                display_frame = draw_detections_with_depth(display_frame, detections, depth_stats_list, thickness=2)
                
                # Overlay disparity if requested
                if SHOW_DISPARITY and frame_count % 3 == 0:  # show every 3 frames to reduce compute
                    try:
                        disp_color = visualize_disparity(disparity)
                        disp_color = cv2.resize(disp_color, (300, 225))
                        display_frame[20:245, 20:320] = disp_color
                    except:
                        pass
                
                # Center crosshair
                if SHOW_CENTER_CROSS:
                    cy, cx = h // 2, w // 2
                    cv2.line(display_frame, (cx - 20, cy), (cx + 20, cy), (255, 0, 0), 1)
                    cv2.line(display_frame, (cx, cy - 20), (cx, cy + 20), (255, 0, 0), 1)
                
                # FPS counter
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_clock
                    display_fps = 30 / elapsed
                    fps_clock = time.time()
                
                cv2.putText(display_frame, f"FPS: {display_fps:.1f}", (w - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Detections: {len(detections)}", (w - 150, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Resize for display
                display_frame = cv2.resize(display_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            else:
                display_frame = cv2.resize(frame_l, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                cv2.putText(display_frame, "PAUSED", (DISPLAY_WIDTH//2 - 80, DISPLAY_HEIGHT//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            
            # Display
            if fullscreen:
                cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            cv2.imshow(window_name, display_frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("[INFO] Exiting...")
                break
            elif key == 32:  # SPACE
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'}")
            elif key == ord('f'):  # f
                fullscreen = not fullscreen
            elif key == ord('r'):  # r
                print("[INFO] Resetting stereo...")
                mapLx, mapLy, mapRx, mapRy, Q, K_L_use = setup_stereo_rectification(w_rect, h_rect, SCALE_FOR_MATCHING)
            
            # Frame rate throttle
            time.sleep(1.0 / TARGET_FPS)
    
    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")

if __name__ == "__main__":
    main()
