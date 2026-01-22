"""
Live camera view with on-demand flower detection and depth capture.

Shows continuous camera feed. Press SPACE to capture current frame,
detect flowers, compute depth, and save results.

Controls:
  SPACE: Capture frame and analyze (saves results to CSV)
  ESC: Exit
  'r': Reset/clear results
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
from datetime import datetime
import csv
import time

# =============================
# Configuration
# =============================
YOLO_MODEL_PATH = os.path.join("..", "current_best_yolo.pt")
CALIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Cam_Params/stereo_charuco_calibration_16cm.npz'))

CAM_LEFT = 1
CAM_RIGHT = 2

CONFIDENCE_THRESHOLD = 0.6
SCALE_FOR_MATCHING = 0.8

MIN_DEPTH = 0.20  # meters
MAX_DEPTH = 0.60  # meters
EXPECTED_DISTANCE = 0.40  # typical object distance for reference

# Display settings
DISPLAY_SCALE = 0.6  # Scale factor for display

# Output file
OUTPUT_CSV = "flower_detection_results.csv"

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
    
    return mapLx, mapLy, mapRx, mapRy, Q


def compute_stereo_disparity(rectL, rectR, Q):
    """Compute disparity and depth map."""
    if rectL.ndim == 3:
        rectL_gray = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        rectR_gray = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    else:
        rectL_gray = rectL
        rectR_gray = rectR
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    rectL_proc = clahe.apply(rectL_gray)
    rectR_proc = clahe.apply(rectR_gray)
    
    fx = abs(Q[2, 3])  # Extract focal length from Q matrix
    expected_disp = (fx * baseline) / EXPECTED_DISTANCE
    num_disp = int(np.ceil(expected_disp * 1.8 / 16.0) * 16)
    num_disp = max(160, min(num_disp, 640))
    
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
    
    return disparity

def compute_depth_map(disparity, Q):
    """Compute depth map from disparity."""
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    depth_map = points_3d[:, :, 2]
    
    # Mask invalid disparities
    invalid_mask = (disparity <= 0) | ~np.isfinite(depth_map)
    depth_map[invalid_mask] = 0.0
    
    return depth_map

def detect_flowers(frame, conf_threshold=0.5):
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
    """Estimate depth for a bounding box ROI."""
    x1, y1, x2, y2 = roi_box
    
    if scale_factor != 1.0:
        x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)
    
    h, w = depth_map.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    roi_depth = depth_map[y1:y2, x1:x2]
    valid_depths = roi_depth[(roi_depth > 0) & (roi_depth >= MIN_DEPTH) & (roi_depth <= MAX_DEPTH)]
    
    if len(valid_depths) == 0:
        return None
    
    median_depth = float(np.median(valid_depths))
    mad = np.median(np.abs(valid_depths - median_depth))
    if mad > 0:
        inlier_mask = np.abs(valid_depths - median_depth) < 3 * mad
        if np.sum(inlier_mask) > 0:
            valid_depths = valid_depths[inlier_mask]
    
    return {
        'min': float(valid_depths.min()),
        'max': float(valid_depths.max()),
        'mean': float(valid_depths.mean()),
        'median': float(np.median(valid_depths)),
        'valid_pixels': len(valid_depths),
        'coverage': 100.0 * len(valid_depths) / roi_depth.size
    }

def save_results_to_csv(timestamp, detections, depth_stats_list):
    """Save detection and depth results to CSV."""
    file_exists = os.path.exists(OUTPUT_CSV)
    
    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['Timestamp', 'Flower_ID', 'X1', 'Y1', 'X2', 'Y2', 
                           'Confidence', 'Depth_Min', 'Depth_Max', 'Depth_Mean', 
                           'Depth_Median', 'Valid_Pixels', 'Coverage_Percent'])
        
        for i, (det, depth_stats) in enumerate(zip(detections, depth_stats_list)):
            x1, y1, x2, y2 = det['box']
            conf = det['confidence']
            
            if depth_stats:
                writer.writerow([
                    timestamp, i+1, x1, y1, x2, y2, f"{conf:.3f}",
                    f"{depth_stats['min']:.4f}", f"{depth_stats['max']:.4f}",
                    f"{depth_stats['mean']:.4f}", f"{depth_stats['median']:.4f}",
                    depth_stats['valid_pixels'], f"{depth_stats['coverage']:.1f}"
                ])
            else:
                writer.writerow([
                    timestamp, i+1, x1, y1, x2, y2, f"{conf:.3f}",
                    'N/A', 'N/A', 'N/A', 'N/A', 0, 0.0
                ])

def main():
    print(f"\n[INFO] Opening cameras {CAM_LEFT} and {CAM_RIGHT}...")
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
    
    # Apply current camera settings
    for cap in [cap_left, cap_right]:
        cap.set(cv2.CAP_PROP_EXPOSURE, -4)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, -5)
    
    w = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Capture resolution: {w}x{h}")
    
    # Setup stereo
    print("[INFO] Setting up stereo rectification...")
    w_rect = int(w * SCALE_FOR_MATCHING)
    h_rect = int(h * SCALE_FOR_MATCHING)
    mapLx, mapLy, mapRx, mapRy, Q = setup_stereo_rectification(w_rect, h_rect, SCALE_FOR_MATCHING)
    
    capture_count = 0
    # Live detection state
    last_detections = []
    last_det_time = 0.0
    DETECT_INTERVAL_SEC = 0.6  # run YOLO every ~0.6s to keep FPS reasonable
    
    print("\n" + "="*70)
    print("LIVE CAPTURE WITH DEPTH")
    print("="*70)
    print("Controls:")
    print("  SPACE - Capture frame and analyze")
    print("  r     - Reset results file")
    print("  ESC   - Exit")
    print(f"\nResults saved to: {OUTPUT_CSV}")
    print("="*70 + "\n")
    
    try:
        while True:
            ret_l, frame_l = cap_left.read()
            ret_r, frame_r = cap_right.read()
            
            if ret_l:
                frame_l = cv2.rotate(frame_l, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if ret_r:
                frame_r = cv2.rotate(frame_r, cv2.ROTATE_90_CLOCKWISE)
            
            if not ret_l or not ret_r:
                print("[ERROR] Failed to read frames")
                break
            
            # Periodic live detection on the left frame
            now = time.time()
            if now - last_det_time >= DETECT_INTERVAL_SEC:
                try:
                    last_detections = detect_flowers(frame_l)
                except Exception as _:
                    last_detections = []
                last_det_time = now

            # Create display
            display = frame_l.copy()
            
            # Add instructions
            cv2.putText(display, "LIVE VIEW - Press SPACE to capture", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, f"Captures: {capture_count}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Detections: {len(last_detections)}", (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(display, "ESC=Exit | r=Reset", (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw last detections on the display
            if last_detections:
                for det in last_detections:
                    x1, y1, x2, y2 = det['box']
                    conf = det['confidence']
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Flower {conf:.2f}"
                    y_text = max(25, y1 - 6)
                    cv2.putText(display, label, (x1, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Resize for display
            display_h = int(h * DISPLAY_SCALE)
            display_w = int(w * DISPLAY_SCALE)
            display = cv2.resize(display, (display_w, display_h))
            
            cv2.imshow("Live Capture", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\n[INFO] Exiting...")
                break
            
            elif key == 32:  # SPACE
                print(f"\n[CAPTURE {capture_count + 1}] Processing...")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                # Process stereo
                frame_l_scaled = cv2.resize(frame_l, (w_rect, h_rect))
                frame_r_scaled = cv2.resize(frame_r, (w_rect, h_rect))
                
                rectL = cv2.remap(frame_l_scaled, mapLx, mapLy, cv2.INTER_LINEAR)
                rectR = cv2.remap(frame_r_scaled, mapRx, mapRy, cv2.INTER_LINEAR)
                
                disparity = compute_stereo_disparity(rectL, rectR, Q)
                depth_map = compute_depth_map(disparity, Q)
                
                # Detect flowers
                detections = detect_flowers(frame_l)
                
                print(f"  Found {len(detections)} flower(s)")
                
                # Compute depth for each detection
                depth_stats_list = []
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det['box']
                    conf = det['confidence']
                    
                    scaled_box = tuple(int(coord * SCALE_FOR_MATCHING) for coord in det['box'])
                    depth_stats = estimate_roi_depth(depth_map, scaled_box)
                    depth_stats_list.append(depth_stats)
                    
                    if depth_stats:
                        print(f"  Flower {i+1}:")
                        print(f"    Position: ({x1}, {y1}) to ({x2}, {y2})")
                        print(f"    Center: ({(x1+x2)//2}, {(y1+y2)//2})")
                        print(f"    Confidence: {conf:.3f}")
                        print(f"    Depth: {depth_stats['median']:.3f}m (median)")
                        print(f"    Depth range: {depth_stats['min']:.3f}m - {depth_stats['max']:.3f}m")
                        print(f"    Coverage: {depth_stats['coverage']:.1f}%")
                    else:
                        print(f"  Flower {i+1}: No valid depth data")
                
                # Save to CSV
                save_results_to_csv(timestamp, detections, depth_stats_list)
                
                capture_count += 1
                print(f"  Results saved to {OUTPUT_CSV}")
            
            elif key == ord('r'):  # Reset
                if os.path.exists(OUTPUT_CSV):
                    os.remove(OUTPUT_CSV)
                    print("\n[INFO] Results file reset")
                capture_count = 0
    
    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Cleanup complete")
        print(f"[INFO] Total captures: {capture_count}")
        if capture_count > 0:
            print(f"[INFO] Results saved in: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
