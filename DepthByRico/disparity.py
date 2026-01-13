import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
# =============================
# Configuration
# =============================

# Set to custom folder path to use captured images, e.g. "stereo_capture_20260109_143022"
# Leave None to use calibration images
CUSTOM_FOLDER = "stereo_capture_20260113_134128"

# Try swapping left/right if depths seem wrong
SWAP_CAMERAS = False

# Downscale before matching to keep disparities within range (0.5 halves disparity ~2x depth range)
SCALE = 0.5

if CUSTOM_FOLDER:
    # Use custom captured images
    CALIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Cam_Params/stereo_charuco_calibration_16cm_rico.npz'))
    if SWAP_CAMERAS:
        LEFT_IMG_PATH = os.path.join(CUSTOM_FOLDER, "right.jpg")   # Swapped
        RIGHT_IMG_PATH = os.path.join(CUSTOM_FOLDER, "left.jpg")   # Swapped
        print(f"Using custom folder (CAMERAS SWAPPED): {CUSTOM_FOLDER}")
    else:
        LEFT_IMG_PATH = os.path.join(CUSTOM_FOLDER, "left.jpg")
        RIGHT_IMG_PATH = os.path.join(CUSTOM_FOLDER, "right.jpg")
        print(f"Using custom folder: {CUSTOM_FOLDER}")
else:
    # Use calibration images from 16CM set
    CALIB_PATH = os.getenv('STEREO_PARAMS_16CM_RICO')
    LEFT_IMG_PATH = os.path.join(os.getenv('CALIB_IMGS_16CM_DIR'), "camera_0_0.jpg")
    RIGHT_IMG_PATH = os.path.join(os.getenv('CALIB_IMGS_16CM_DIR'), "camera_2_0.jpg")
    print("Using calibration images (16CM)")

OUTPUT_DIR = "stereo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# Load calibration
# =============================
calib = np.load(CALIB_PATH)
K_L = calib["K_left"]
D_L = calib["D_left"]
K_R = calib["K_right"]
D_R = calib["D_right"]
R = calib["R"]
T = calib["T"]

# =============================
# Load images
# =============================
imgL = cv2.imread(LEFT_IMG_PATH)
imgR = cv2.imread(RIGHT_IMG_PATH)

if imgL is None or imgR is None:
    raise IOError("Could not load stereo images")

# Flip images if cameras are upside down
FLIP_IMAGES = False
if FLIP_IMAGES:
    imgL = cv2.flip(imgL, -1)  # -1 = flip both horizontally and vertically (180 degrees)
    imgR = cv2.flip(imgR, -1)
    print("[INFO] Images flipped 180 degrees")

orig_imgL = imgL.copy()
orig_imgR = imgR.copy()

h_full, w_full = imgL.shape[:2]

def scale_intrinsics(K, scale):
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale
    K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale
    K_scaled[1, 2] *= scale
    return K_scaled

scale = float(SCALE)
if scale != 1.0:
    print(f"[INFO] Downscaling images by {scale:.2f}x for disparity range")
    w = int(w_full * scale)
    h = int(h_full * scale)
    imgL = cv2.resize(imgL, (w, h))
    imgR = cv2.resize(imgR, (w, h))
else:
    w, h = w_full, h_full

K_L_use = scale_intrinsics(K_L, scale)
K_R_use = scale_intrinsics(K_R, scale)

# =============================
# Stereo rectification
# =============================
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K_L_use, D_L, K_R_use, D_R, (w, h), R, T, alpha=1
)

mapLx, mapLy = cv2.initUndistortRectifyMap(
    K_L_use, D_L, R1, P1, (w, h), cv2.CV_32FC1
)
mapRx, mapRy = cv2.initUndistortRectifyMap(
    K_R_use, D_R, R2, P2, (w, h), cv2.CV_32FC1
)

rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

# Save rectified left image (YOLO must use this)
cv2.imwrite(os.path.join(OUTPUT_DIR, "rectified_left.png"), rectL)
cv2.imwrite(os.path.join(OUTPUT_DIR, "rectified_right.png"), rectR)

# Improve contrast to help matcher (especially on low-texture scenes)
if rectL.ndim == 3:
    rectL_gray = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    rectR_gray = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
else:
    rectL_gray = rectL
    rectR_gray = rectR

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
rectL_proc = clahe.apply(rectL_gray)
rectR_proc = clahe.apply(rectR_gray)

# =============================
# Disparity computation
# =============================
expected_disp_40cm = (K_L_use[0, 0] * np.linalg.norm(T)) / 0.40
num_disp = int(np.ceil(expected_disp_40cm * 2.0 / 16.0) * 16)
num_disp = max(160, min(num_disp, 640))  # clamp to keep compute reasonable
block_size = 5

print(f"[INFO] Expected disparity at 0.40m (scaled): {expected_disp_40cm:.1f}px; using search {num_disp}px")

stereo = cv2.StereoSGBM.create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * block_size**2,
    P2=32 * block_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=6,
    speckleWindowSize=80,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

disparity = stereo.compute(rectL_proc, rectR_proc).astype(np.float32) / 16.0

# =============================
# Depth computation (Z only)
# =============================
points_3d = cv2.reprojectImageTo3D(disparity, Q)
depth_map = points_3d[:, :, 2]  # Z-coordinate

# Mask invalid disparities (zero or negative) and non-finite depths
invalid_mask = (disparity <= 0) | ~np.isfinite(depth_map)
depth_map[invalid_mask] = 0.0

# =============================
# Depth Statistics
# =============================
valid_disp_mask = disparity > 0
if np.any(valid_disp_mask):
    disp_valid = disparity[valid_disp_mask]
    print(f"[STATS] Disparity valid: {disp_valid.size} / {disparity.size} ({100*disp_valid.size/disparity.size:.1f}%)")
    print(f"[STATS] Disparity range: {disp_valid.min():.2f}px - {disp_valid.max():.2f}px (median {np.median(disp_valid):.2f}px)")
else:
    print("[WARNING] No valid disparity values found!")

valid_depths = depth_map[depth_map > 0]
if len(valid_depths) > 0:
    print(f"[STATS] Valid depth pixels: {len(valid_depths)} / {depth_map.size} ({100*len(valid_depths)/depth_map.size:.1f}%)")
    print(f"[STATS] Depth range: {valid_depths.min():.4f}m - {valid_depths.max():.4f}m")
    print(f"[STATS] Median depth: {np.median(valid_depths):.4f}m")
    print(f"[STATS] Mean depth: {valid_depths.mean():.4f}m")
    
    # Sample regions
    h, w = depth_map.shape
    cy, cx = h//2, w//2
    roi_size = 50
    roi = depth_map[max(0,cy-roi_size):min(h,cy+roi_size), max(0,cx-roi_size):min(w,cx+roi_size)]
    roi_valid = roi[roi > 0]
    if len(roi_valid) > 0:
        print(f"[STATS] Center region (±{roi_size}px): median={np.median(roi_valid):.4f}m, mean={roi_valid.mean():.4f}m")
else:
    print("[WARNING] No valid depth values found!")

# =============================
# Save numerical outputs
# =============================
np.save(os.path.join(OUTPUT_DIR, "disparity.npy"), disparity)
np.save(os.path.join(OUTPUT_DIR, "depth_map.npy"), depth_map)

print("[INFO] Saved disparity.npy and depth_map.npy")

# =============================
# Visualization (sanity check)
# =============================

# Disparity visualization
disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp_vis = disp_vis.astype(np.uint8)
disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# Depth visualization - only use valid depths for normalization
depth_for_vis = depth_map.copy()
valid_depths_mask = depth_for_vis > 0
if np.any(valid_depths_mask):
    # Normalize using only valid depths
    valid_min = valid_depths_mask * depth_for_vis
    valid_min = valid_min[valid_min > 0].min()
    valid_max = depth_for_vis[valid_depths_mask].max()
    
    # Clip outliers (e.g., very far pixels)
    depth_clipped = depth_for_vis.copy()
    depth_clipped[depth_clipped > valid_max * 0.9] = valid_max * 0.9  # Remove extreme outliers
    
    depth_vis = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "depth_map_color.png"), depth_color)
    print("[INFO] Saved depth_map_color.png")

def resize_for_display(img, width=900, flip=False):
    h, w = img.shape[:2]
    scale = width / w
    resized = cv2.resize(img, (width, int(h * scale)))
    if flip:
        resized = cv2.flip(resized, -1)  # Flip 180 degrees for display only
    return resized

# Set to True if camera display is upside down
FLIP_DISPLAY = True

# Prepare images for composite display - resize to uniform size
display_size = 400  # pixels per tile (reduced from 800)

def prep_display_img(img, size, flip=False):
    """Resize image to square and optionally flip"""
    if flip:
        img = cv2.flip(img, -1)
    h, w = img.shape[:2]
    # Resize to square, maintaining aspect ratio
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    # Pad to exact square size
    padded = np.zeros((size, size, 3 if len(img.shape) == 3 else 1), dtype=img.dtype)
    start_y = (size - new_h) // 2
    start_x = (size - new_w) // 2
    padded[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    return padded

# Prepare all images
img_L_display = prep_display_img(orig_imgL, display_size, flip=FLIP_DISPLAY)
img_R_display = prep_display_img(orig_imgR, display_size, flip=FLIP_DISPLAY)
disp_display = prep_display_img(disp_color, display_size, flip=FLIP_DISPLAY)

if np.any(valid_depths_mask):
    depth_display = prep_display_img(depth_color, display_size, flip=FLIP_DISPLAY)
else:
    depth_display = np.zeros((display_size, display_size, 3), dtype=np.uint8)

# Create 2x2 grid: top row = originals, bottom row = maps
top_row = np.hstack([img_L_display, img_R_display])
bottom_row = np.hstack([disp_display, depth_display])
composite = np.vstack([top_row, bottom_row])

# Save all diagnostic images
cv2.imwrite(os.path.join(OUTPUT_DIR, "orig_left.png"), orig_imgL)
cv2.imwrite(os.path.join(OUTPUT_DIR, "orig_right.png"), orig_imgR)
cv2.imwrite(os.path.join(OUTPUT_DIR, "disp_color.png"), disp_color)
if np.any(valid_depths_mask):
    cv2.imwrite(os.path.join(OUTPUT_DIR, "depth_map_color.png"), depth_color)
cv2.imwrite(os.path.join(OUTPUT_DIR, "composite_grid.png"), composite)

print("\nDisplaying composite: Press any key to continue...")
cv2.imshow("Stereo Analysis", composite)
cv2.waitKey(0)

cv2.destroyAllWindows()
