import cv2
import numpy as np
import glob
import os

# ============================================================
# =================== USER CONFIG ============================
# ============================================================

import os

# Get the parent directory (BiomimicryCode)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEFT_PATH  = os.path.join(BASE_DIR, "CalibImg/ArucoImages16cm/camera_1_*.jpg")
RIGHT_PATH = os.path.join(BASE_DIR, "CalibImg/ArucoImages16cm/camera_2_*.jpg")
MIN_CORNERS = 15

# ==== Charuco board definition ====
aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

board = cv2.aruco.CharucoBoard(
    (11, 8),      # squaresX, squaresY
    0.035,        # square length (meters)
    0.026,        # marker length (meters)
    aruco_dict
)

board.setLegacyPattern(True)

# ============================================================
# =================== DETECTORS ==============================
# ============================================================

detector_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

charuco_params = cv2.aruco.CharucoParameters()
charuco_detector = cv2.aruco.CharucoDetector(
    board,
    charuco_params,
    detector_params
)

board.setLegacyPattern(True)


# ============================================================
# =================== LOAD IMAGES ============================
# ============================================================

left_images  = sorted(glob.glob(LEFT_PATH))
right_images = sorted(glob.glob(RIGHT_PATH))

assert len(left_images) == len(right_images)
assert len(left_images) > 0

print(f"[INFO] Found {len(left_images)} image pairs")

# ============================================================
# ================= CORNER DETECTION =========================
# ============================================================

charuco_corners_l, charuco_ids_l = [], []
charuco_corners_r, charuco_ids_r = [], []

image_size = None
valid_pairs = 0

for lp, rp in zip(left_images, right_images):
    img_l = cv2.imread(lp)
    img_r = cv2.imread(rp)

    if img_l is None or img_r is None:
        continue

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray_l.shape[::-1]

    # ---- LEFT ----
    ch_corners_l, ch_ids_l, _, _ = charuco_detector.detectBoard(gray_l)

    # ---- RIGHT ----
    ch_corners_r, ch_ids_r, _, _ = charuco_detector.detectBoard(gray_r)

    if ch_ids_l is None or ch_ids_r is None:
        continue

    n_l = len(ch_ids_l)
    n_r = len(ch_ids_r)

    print(f"[DEBUG] corners L={n_l}, R={n_r}")

    if n_l >= MIN_CORNERS and n_r >= MIN_CORNERS:
        charuco_corners_l.append(ch_corners_l)
        charuco_ids_l.append(ch_ids_l)

        charuco_corners_r.append(ch_corners_r)
        charuco_ids_r.append(ch_ids_r)

        valid_pairs += 1


print(f"[INFO] Valid stereo pairs: {valid_pairs}")
assert valid_pairs >= 10

# ============================================================
# ================= MONO CALIBRATION =========================
# ============================================================

obj_points_l = []
img_points_l = []

for corners, ids in zip(charuco_corners_l, charuco_ids_l):
    objp, imgp = board.matchImagePoints(corners, ids)
    obj_points_l.append(objp)
    img_points_l.append(imgp)

print("[INFO] Calibrating LEFT camera...")

err_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
    objectPoints=obj_points_l,
    imagePoints=img_points_l,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

print(f"[RESULT] Left reprojection error: {err_l:.4f}")

obj_points_r = []
img_points_r = []

for corners, ids in zip(charuco_corners_r, charuco_ids_r):
    objp, imgp = board.matchImagePoints(corners, ids)
    obj_points_r.append(objp)
    img_points_r.append(imgp)

print("[INFO] Calibrating RIGHT camera...")

err_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
    objectPoints=obj_points_r,
    imagePoints=img_points_r,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

print(f"[RESULT] Right reprojection error: {err_r:.4f}")


# ============================================================
# ================== STEREO CALIBRATION ======================
# ============================================================

obj_points_stereo = []
img_points_l_stereo = []
img_points_r_stereo = []

for corners_l, ids_l, corners_r, ids_r in zip(
    charuco_corners_l, charuco_ids_l,
    charuco_corners_r, charuco_ids_r
):
    # Find common Charuco corner IDs
    ids_l_flat = ids_l.flatten()
    ids_r_flat = ids_r.flatten()

    common_ids = np.intersect1d(ids_l_flat, ids_r_flat)

    if len(common_ids) < 6:
        continue

    # Select matching corners
    mask_l = np.isin(ids_l_flat, common_ids)
    mask_r = np.isin(ids_r_flat, common_ids)

    matched_corners_l = corners_l[mask_l]
    matched_corners_r = corners_r[mask_r]
    matched_ids = common_ids.reshape(-1, 1)

    # Convert to object/image points
    objp, imgp_l = board.matchImagePoints(
        matched_corners_l, matched_ids
    )
    _, imgp_r = board.matchImagePoints(
        matched_corners_r, matched_ids
    )

    obj_points_stereo.append(objp)
    img_points_l_stereo.append(imgp_l)
    img_points_r_stereo.append(imgp_r)

print("[INFO] Stereo views used:", len(obj_points_stereo))

stereo_err, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objectPoints=obj_points_stereo,
    imagePoints1=img_points_l_stereo,
    imagePoints2=img_points_r_stereo,
    cameraMatrix1=K_l,
    distCoeffs1=D_l,
    cameraMatrix2=K_r,
    distCoeffs2=D_r,
    imageSize=image_size,
    flags=cv2.CALIB_FIX_INTRINSIC,
    criteria=(
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6
    )
)

baseline = np.linalg.norm(T)

print(f"[RESULT] Stereo reprojection error: {stereo_err:.4f} px")
print(f"[RESULT] Stereo baseline: {baseline:.4f} m")
print("[RESULT] Translation vector T:\n", T)

# ============================================================
# ===================== SAVE RESULT ==========================
# ============================================================

np.savez(
    os.path.join(BASE_DIR, "Cam_Params/stereo_charuco_calibration_16cm.npz"),
    K_left=K_l, D_left=D_l,
    K_right=K_r, D_right=D_r,
    R=R, T=T,
    image_size=image_size
)

print("[INFO] Calibration saved.")
