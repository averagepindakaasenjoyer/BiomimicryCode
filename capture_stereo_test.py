"""
Capture stereo image pair from live cameras for testing depth estimation.
This captures one frame from camera 0 and camera 2, saves them, and shows depth estimation.
"""
import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

LEFT_CAM = 0
RIGHT_CAM = 2
OUTPUT_DIR = "test_captures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Capturing stereo images from cameras...")
print(f"Camera indices: LEFT={LEFT_CAM}, RIGHT={RIGHT_CAM}")
print("Position your test target (ArUco board) at 50cm from the cameras.")
print("Press SPACE to capture, ESC to cancel.")

# Open cameras
cap_left = cv2.VideoCapture(LEFT_CAM)
cap_right = cv2.VideoCapture(RIGHT_CAM)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("ERROR: Could not open cameras!")
    exit(1)

# Set resolution if possible
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

captured = False
while True:
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()
    
    if not ret_l or not ret_r:
        print("ERROR: Could not read from cameras!")
        break
    
    # Display side by side
    h, w = frame_l.shape[:2]
    display = np.hstack([frame_l, frame_r])
    display = cv2.resize(display, (1920, 540))
    
    cv2.putText(display, "LEFT (Camera 0)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display, "RIGHT (Camera 2)", (960+50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display, "SPACE=Capture  ESC=Cancel", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("Stereo Capture - Position target at 50cm", display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # SPACE
        print("\nCapturing...")
        cv2.imwrite(os.path.join(OUTPUT_DIR, "test_left.jpg"), frame_l)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "test_right.jpg"), frame_r)
        print(f"Saved: {OUTPUT_DIR}/test_left.jpg, {OUTPUT_DIR}/test_right.jpg")
        captured = True
        break
    elif key == 27:  # ESC
        print("Cancelled.")
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

if captured:
    print("\nNow run: python Calibration/DepthEstimate.py with SELECTED_IMG_COUPLE pointing to test_captures")
