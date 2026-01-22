import cv2
import numpy as np
# import time
from ultralytics import YOLO
# from collections import defaultdict
import os
import csv
# from datetime import datetime


# ==========================================================
# CONFIG
# ==========================================================
MODEL_PATH = os.path.join("..", "current_best_yolo.pt")
CALIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Cam_Params/stereo_charuco_calibration_16cm.npz'))

OUTPUT_DIR = "stereo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAM_LEFT = 1
CAM_RIGHT = 2

CONFIDENCE_THRESHOLD = 0.6

# Performance
TARGET_FPS = 5
SCALE_FOR_MATCHING = 0.5

# Depth constraints
EXPECTED_DISTANCE = 0.20  # meters
MIN_DEPTH = 0.10
MAX_DEPTH = 2.0

CAPTURE_FRAMES = 5

# ==========================================================
# LOAD CALIBRATION
# ==========================================================
print("[INFO] Loading calibration...")
calib = np.load(CALIB_PATH)

K_L = calib["K_left"]
D_L = calib["D_left"]
K_R = calib["K_right"]
D_R = calib["D_right"]
R = calib["R"]
T = calib["T"]

baseline = np.linalg.norm(T)
print(f"[INFO] Stereo baseline: {baseline:.4f} m")

# ==========================================================
# CAMERA → ROBOT BASE TRANSFORM (FILL WITH REAL VALUES)
# ==========================================================
R_BASE_CAMERA = np.array([
    [ 1,  0,  0],
    [ 0,  1,  0],
    [ 0,  0,  1]
], dtype=np.float64)
# TODO: replace with correct rotation matrix

# --- Translation: Camera origin in robot base frame (meters) ---
t_BASE_CAMERA = np.array([
    [0.0],  # x offset
    [0.0],  # y offset
    [0.0]   # z offset
], dtype=np.float64)
# TODO: measure camera position relative to robot base

TCP_OFFSET = np.array([
    [0.0],   # x
    [0.0],   # y
    [0.12]   # z (example: 12 cm gripper length)
], dtype=np.float64)
# TODO: measure real gripper geometry

# ==========================================================
# DEPTH BIAS MODEL
# ==========================================================
MIN_COVERAGE_PCT = 25.0
MAX_VALID_DEPTH = 0.55  # meters

DEPTH_BIAS_A = 0.0025
DEPTH_BIAS_B = -0.02
DEPTH_BIAS_C = 1.1

def apply_depth_bias(Z_cm):
    bias = DEPTH_BIAS_A * Z_cm**2 + DEPTH_BIAS_B * Z_cm + DEPTH_BIAS_C
    return Z_cm - bias

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def scale_intrinsics(K, scale):
    K2 = K.copy()
    K2[0, 0] *= scale
    K2[1, 1] *= scale
    K2[0, 2] *= scale
    K2[1, 2] *= scale
    return K2

def setup_stereo_rectification(w, h):
    K_L_s = scale_intrinsics(K_L, SCALE_FOR_MATCHING)
    K_R_s = scale_intrinsics(K_R, SCALE_FOR_MATCHING)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_L_s, D_L, K_R_s, D_R, (w, h), R, T, alpha=1
    )

    mapLx, mapLy = cv2.initUndistortRectifyMap(
        K_L_s, D_L, R1, P1, (w, h), cv2.CV_32FC1
    )
    mapRx, mapRy = cv2.initUndistortRectifyMap(
        K_R_s, D_R, R2, P2, (w, h), cv2.CV_32FC1
    )

    return mapLx, mapLy, mapRx, mapRy, Q, K_L_s

def compute_stereo_disparity(rectL, rectR, K_L_s):
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(3.0, (8, 8))
    grayL = clahe.apply(grayL)
    grayR = clahe.apply(grayR)

    # expected_disp = (K_L_s[0, 0] * baseline) / EXPECTED_DISTANCE
    # num_disp = int(np.ceil(expected_disp * 1.8 / 16) * 16)
    # num_disp = max(160, min(num_disp, 640))

    num_disp = 528

    stereo = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=5,
        P1=8 * 5**2,
        P2=32 * 5**2,
        uniquenessRatio=6,
        speckleWindowSize=80,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    return stereo.compute(grayL, grayR).astype(np.float32) / 16.0, num_disp

def estimate_roi_depth(depth_map, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = depth_map.shape

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = depth_map[y1:y2, x1:x2]
    valid = roi[(roi > MIN_DEPTH) & (roi < MAX_DEPTH) & np.isfinite(roi)]

    if len(valid) == 0:
        return None

    return float(np.median(valid)), 100.0 * len(valid) / roi.size

def pixel_to_camera_xyz(u, v, Z):
    """
    Convert pixel + depth → camera-frame 3D point
    """
    fx = K_L[0, 0]
    fy = K_L[1, 1]
    cx = K_L[0, 2]
    cy = K_L[1, 2]

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return np.array([[X], [Y], [Z]])


def camera_to_robot_base(X_cam):
    """
    Transform point from camera frame to robot base frame
    """
    return R_BASE_CAMERA @ X_cam + t_BASE_CAMERA

def compute_grasp_pose(u, v, Z, coverage):
    """
    Main entry point:
    Pixel detection → robot grasp position
    """

    # ---- Validate depth ----
    if coverage < MIN_COVERAGE_PCT:
        return None, "Rejected: low depth coverage"

    if Z > MAX_VALID_DEPTH:
        return None, "Rejected: depth too far"

    # ---- Bias correction ----
    Z_corr = apply_depth_bias(Z * 100)  # Convert meters → cm
    if Z_corr is None:
        return None, "Rejected: low depth coverage"
    Z_corr = Z_corr / 100.0  # Back to meters for XYZ computation

    # ---- Pixel → Camera XYZ ----
    P_cam = pixel_to_camera_xyz(u, v, Z_corr)


    # ---- Camera → Robot Base ----
    P_base = camera_to_robot_base(P_cam)

    # ---- Add gripper TCP offset ----
    P_grasp = P_base - TCP_OFFSET

    return P_grasp, "OK"

# Overlay
def draw_detection_overlay(frame, box, Z, coverage):
    x1, y1, x2, y2 = map(int, box)
    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Center point
    cv2.circle(frame, (u, v), 4, (0, 0, 255), -1)

    # Text
    label = f"(u,v)=({u},{v})  Z={Z:.3f}m  cov={coverage:.0f}%"
    cv2.putText(
        frame,
        label,
        (x1, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


# ==========================================================
# CSV LOGGING SETUP
# ==========================================================

def create_numbered_csv(log_folder, log_id):
    filename = f"log_{log_id:03d}.csv"
    path = os.path.join(log_folder, filename)

    file = open(path, "w", newline="")
    writer = csv.writer(file)

    writer.writerow([
        "log_id",
        "frame_idx",
        "u_pixel",
        "v_pixel",
        "depth_m",
        "coverage_pct",
        "X_base_m",
        "Y_base_m",
        "Z_base_m"
    ])

    return file, writer


def create_log_folder(base_dir, prefix="log"):
    existing = [f for f in os.listdir(base_dir) if f.startswith(prefix)]
    next_id = len(existing) + 1
    folder_name = f"{prefix}_{next_id:03d}"
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    return path, next_id


# ==========================================================
# MAIN
# ==========================================================
def main():
    capture_done = False

    # Use DirectShow backend to avoid MSMF errors
    capL = cv2.VideoCapture(CAM_LEFT + cv2.CAP_DSHOW)
    capR = cv2.VideoCapture(CAM_RIGHT + cv2.CAP_DSHOW)

    if not capL.isOpened() or not capR.isOpened():
        print("[ERROR] Could not open cameras!")
        return

    capL.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Apply camera settings for proper exposure
    for cap in [capL, capR]:
        cap.set(cv2.CAP_PROP_EXPOSURE, -4)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, -5)

    model = YOLO(MODEL_PATH)

    ret, frameL = capL.read()
    h, w = frameL.shape[:2]
    w_s, h_s = int(w * SCALE_FOR_MATCHING), int(h * SCALE_FOR_MATCHING)

    mapLx, mapLy, mapRx, mapRy, Q, K_L_s = setup_stereo_rectification(w_s, h_s)

    frame_counter = 0
    image_counter = 0
    pending_capture = False
    detections = []

    print("[INFO] Press ENTER to capture 5 frames")

    # Create log folder and CSV
    log_folder, log_id = create_log_folder(OUTPUT_DIR)
    csv_file, csv_writer = create_numbered_csv(log_folder, log_id)
    print(f"[INFO] Logging to folder: {log_folder}")

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            break

        # Rotate frames to align cameras (opposite 90 degree rotations)
        if retL:
            frameL = cv2.rotate(frameL, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if retR:
            frameR = cv2.rotate(frameR, cv2.ROTATE_90_CLOCKWISE)

        frame_counter += 1

        frameL_s = cv2.resize(frameL, (w_s, h_s))
        frameR_s = cv2.resize(frameR, (w_s, h_s))
        rectL = cv2.remap(frameL_s, mapLx, mapLy, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR_s, mapRx, mapRy, cv2.INTER_LINEAR)

        disparity, _ = compute_stereo_disparity(rectL, rectR, K_L_s)
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        depth_map = points_3d[:, :, 2]
        depth_map[(disparity <= 0) | ~np.isfinite(depth_map)] = 0

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and not pending_capture:
            pending_capture = True
            capture_done = False
            detections.clear()
            image_counter = 0
            print("\n[CAPTURE STARTED]")

        # --- ALWAYS run YOLO for live visualization ---
        results = model.predict(frameL, conf=0.35, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                if float(box.conf[0]) < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                box_s = (
                    x1 * SCALE_FOR_MATCHING,
                    y1 * SCALE_FOR_MATCHING,
                    x2 * SCALE_FOR_MATCHING,
                    y2 * SCALE_FOR_MATCHING
                )

                depth_info = estimate_roi_depth(depth_map, box_s)
                if depth_info is None:
                    continue

                Z, coverage = depth_info

                draw_detection_overlay(
                    frameL,
                    (x1, y1, x2, y2),
                    Z,
                    coverage
                )

                if pending_capture:
                    detections.append((x1, y1, x2, y2, Z, coverage))
                    print(
                        f"[Frame {frame_counter}] "
                        f"YOLO=({(x1 + x2) / 2:.1f},{(y1 + y2) / 2:.1f}) "
                        f"Z={Z:.3f}m"
                    )
                    # Save frame as PNG
                    image_path = os.path.join(log_folder, f"frame_{image_counter:03d}.png")
                    cv2.imwrite(image_path, frameL)
                    image_counter += 1

            if pending_capture and len(detections) >= CAPTURE_FRAMES and not capture_done:
                pending_capture = False
                capture_done = True
                chosen = max(detections, key=lambda d: d[5])

                x1, y1, x2, y2, Z, coverage = chosen
                u, v = (x1 + x2) / 2, (y1 + y2) / 2

                Z_corr = apply_depth_bias(Z * 100) / 100.0
                P_base, status = compute_grasp_pose(u, v, Z_corr, coverage)

                if P_base is None:
                    print(f"[REJECTED] {status}")
                    continue

                P_base = P_base.flatten()

                csv_writer.writerow([
                    log_id,
                    frame_counter,
                    round(u, 2),
                    round(v, 2),
                    round(Z_corr, 4),
                    round(coverage, 2),
                    round(float(P_base[0]), 4),
                    round(float(P_base[1]), 4),
                    round(float(P_base[2]), 4),
                ])

                csv_file.flush()

                print("\n=== FINAL OUTPUT ===")
                print(f"Pixel (u,v): ({u:.1f},{v:.1f})")
                print(f"Depth: {Z_corr:.3f} m")
                print("Robot base XYZ (m):")
                print(P_base)

                detections.clear()

        cv2.imshow("Live YOLO + Depth", frameL)
        if key == 27:
            break

    csv_file.close()
    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
