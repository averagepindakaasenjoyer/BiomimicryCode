import cv2
import os
import numpy as np
from datetime import datetime

# ============================================================
# =================== USER CONFIG ============================
# ============================================================

CAMERA_LEFT_INDEX = 1
CAMERA_RIGHT_INDEX = 2

# Get the parent directory (BiomimicryCode)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "CalibImg", "ArucoImages16cm")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INFO] Output directory: {OUTPUT_DIR}")
print(f"[INFO] Opening cameras at index {CAMERA_LEFT_INDEX} and {CAMERA_RIGHT_INDEX}")

# ============================================================
# =================== OPEN CAMERAS ===========================
# ============================================================

cap_left = cv2.VideoCapture(CAMERA_LEFT_INDEX)
cap_right = cv2.VideoCapture(CAMERA_RIGHT_INDEX)

if not cap_left.isOpened():
    print(f"[ERROR] Could not open camera {CAMERA_LEFT_INDEX}")
    exit(1)

if not cap_right.isOpened():
    print(f"[ERROR] Could not open camera {CAMERA_RIGHT_INDEX}")
    cap_left.release()
    exit(1)

print("[INFO] Cameras opened successfully")

# Set resolution (optional - adjust as needed)
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Counter for saved images
image_count = 0

print("\n[INFO] Press SPACEBAR to capture image pair")
print("[INFO] Press ESC or 'q' to quit")
print(f"[INFO] Images will be saved as camera_1_X.jpg and camera_2_X.jpg")

# ============================================================
# =================== CAPTURE LOOP ===========================
# ============================================================

while True:
    # Read frames from both cameras
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left or not ret_right:
        print("[ERROR] Failed to read frames from cameras")
        break
    
    # Rotate cameras to align them (opposite 90 degree rotations)
    frame_left = cv2.rotate(frame_left, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame_right = cv2.rotate(frame_right, cv2.ROTATE_90_CLOCKWISE)
    
    # Display both frames side by side
    combined = np.hstack((frame_left, frame_right))
    cv2.imshow('Stereo Calibration Capture (Left | Right)', combined)
    
    key = cv2.waitKey(1) & 0xFF
    
    # SPACEBAR - Capture image
    if key == 32:  # Spacebar
        left_filename = f"camera_1_{image_count}.jpg"
        right_filename = f"camera_2_{image_count}.jpg"
        
        left_path = os.path.join(OUTPUT_DIR, left_filename)
        right_path = os.path.join(OUTPUT_DIR, right_filename)
        
        cv2.imwrite(left_path, frame_left)
        cv2.imwrite(right_path, frame_right)
        
        print(f"[SAVED] Image pair {image_count}: {left_filename}, {right_filename}")
        image_count += 1
    
    # ESC or 'q' - Quit
    elif key == 27 or key == ord('q'):
        break

# ============================================================
# =================== CLEANUP ================================
# ============================================================

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

print(f"\n[INFO] Capture complete. Total image pairs saved: {image_count}")
print(f"[INFO] Images saved in: {OUTPUT_DIR}")
