"""
Capture one frame from each USB camera and save to a folder.
"""
import cv2
import os
from datetime import datetime

# Camera indices - adjust if needed
CAM_LEFT = 1
CAM_RIGHT = 2  # or 1 depending on your setup

# Create output folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"stereo_capture_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print(f"Output folder: {output_dir}")
print(f"Opening cameras {CAM_LEFT} and {CAM_RIGHT}...")

# Open cameras
cap_left = cv2.VideoCapture(CAM_LEFT)
cap_right = cv2.VideoCapture(CAM_RIGHT)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("ERROR: Could not open one or both cameras!")
    exit(1)

# Optionally set resolution
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Capturing frames... Press SPACE to capture or ESC to cancel")

while True:
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()
    
    if not ret_l or not ret_r:
        print("ERROR reading frames")
        break
    
    # Display preview
    h, w = frame_l.shape[:2]
    preview = cv2.hconcat([frame_l, frame_r])
    preview = cv2.resize(preview, (1920, 540))
    
    cv2.putText(preview, f"CAM {CAM_LEFT} (LEFT)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(preview, f"CAM {CAM_RIGHT} (RIGHT)", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(preview, "SPACE=Capture  ESC=Cancel", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("Stereo Capture", preview)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # SPACE
        left_path = os.path.join(output_dir, "left.jpg")
        right_path = os.path.join(output_dir, "right.jpg")
        
        cv2.imwrite(left_path, frame_l)
        cv2.imwrite(right_path, frame_r)
        
        print(f"\n✓ Captured!")
        print(f"  Left:  {left_path}")
        print(f"  Right: {right_path}")
        print(f"\nTo use in disparity.py, set:")
        print(f"  CUSTOM_FOLDER = '{output_dir}'")
        break
    elif key == 27:  # ESC
        print("Cancelled")
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
