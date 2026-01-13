import os
import cv2
import numpy as np
from ultralytics import YOLO

# =============================
# Paths
# =============================
MODEL_PATH = "runs/detect/train17/weights/best.pt"
RECT_LEFT_IMG = "stereo_output/rectified_left.png"
DEPTH_MAP_PATH = "stereo_output/depth_map.npy"
CALIB_PATH = "stereo_charuco_calibration.npz"

OUTPUT_DIR = "fusion_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# Load data
# =============================
model = YOLO(MODEL_PATH)

img = cv2.imread(RECT_LEFT_IMG)
if img is None:
    raise IOError("Could not load rectified_left.png")

depth_map = np.load(DEPTH_MAP_PATH)

calib = np.load(CALIB_PATH)
K = calib["K_left"]
cx = K[0, 2]
cy = K[1, 2]
f = K[0, 0]  # focal length in pixels (used only for X,Y)

h, w = img.shape[:2]

# =============================
# Run YOLO
# =============================
results = model.predict(
    source=img,
    conf=0.35,
    save=False,
    verbose=False
)

detections = []

# =============================
# Parameters
# =============================
ROI_RADIUS = 5  # 11x11 window

# =============================
# Process detections
# =============================
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        # Center of bounding box
        cx_img = (x1 + x2) / 2
        cy_img = (y1 + y2) / 2

        u = int(round(cx_img))
        v = int(round(cy_img))

        # Bounds check
        if u < ROI_RADIUS or v < ROI_RADIUS or u >= w - ROI_RADIUS or v >= h - ROI_RADIUS:
            continue

        roi = depth_map[
            v - ROI_RADIUS : v + ROI_RADIUS + 1,
            u - ROI_RADIUS : u + ROI_RADIUS + 1
        ]

        valid = roi[np.isfinite(roi) & (roi > 0)]

        if len(valid) == 0:
            continue

        Z = float(np.median(valid))

        # Back-project to camera coordinates
        X = (u - cx) * Z / f
        Y = (v - cy) * Z / f

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "center_px": (cx_img, cy_img),
            "xyz": (X, Y, Z),
            "confidence": conf,
            "class": cls
        })

        # Draw on image
        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        cv2.circle(img, (u, v), 4, (0, 0, 255), -1)

        cv2.putText(
            img,
            f"Z={Z:.2f}m",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

# =============================
# Save results
# =============================
img_out_path = os.path.join(OUTPUT_DIR, "fusion_result.png")
cv2.imwrite(img_out_path, img)

txt_out_path = os.path.join(OUTPUT_DIR, "detections_xyz.txt")
with open(txt_out_path, "w") as f:
    for i, det in enumerate(detections, 1):
        X, Y, Z = det["xyz"]
        cx_img, cy_img = det["center_px"]
        f.write(
            f"Detection {i}:\n"
            f"  Center (px): ({cx_img:.2f}, {cy_img:.2f})\n"
            f"  X = {X:.4f} m\n"
            f"  Y = {Y:.4f} m\n"
            f"  Z = {Z:.4f} m\n\n"
        )

print(f"[INFO] Saved annotated image to {img_out_path}")
print(f"[INFO] Saved XYZ detections to {txt_out_path}")

# =============================
# Display
# =============================
def resize_for_display(img, width=900):
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)))

cv2.imshow("YOLO + Stereo Fusion", resize_for_display(img))
cv2.waitKey(0)
cv2.destroyAllWindows()
