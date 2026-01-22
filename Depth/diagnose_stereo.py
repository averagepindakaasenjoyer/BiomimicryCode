"""
Diagnostic script to check stereo calibration and depth accuracy.
"""
import cv2
import numpy as np
import os

# Load calibration
CALIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Cam_Params/stereo_charuco_calibration_16cm.npz'))

print("=" * 60)
print("STEREO CALIBRATION DIAGNOSTIC")
print("=" * 60)

calib = np.load(CALIB_PATH)
K_L = calib["K_left"]
D_L = calib["D_left"]
K_R = calib["K_right"]
D_R = calib["D_right"]
R = calib["R"]
T = calib["T"]

baseline = np.linalg.norm(T)
fx_left = K_L[0, 0]
fx_right = K_R[0, 0]
cx_left = K_L[0, 2]
cx_right = K_R[0, 2]

print("\n1. BASELINE (Distance between cameras)")
print(f"   Measured: {baseline:.6f} m ({baseline*100:.2f} cm)")
print(f"   Expected: ~0.16 m (16 cm)")
print(f"   ⚠️  ERROR: {abs(baseline - 0.16)*100:.1f} cm difference" if abs(baseline - 0.16) > 0.01 else "   ✓ OK")

print("\n2. FOCAL LENGTHS")
print(f"   Left camera:  fx = {fx_left:.2f} pixels")
print(f"   Right camera: fx = {fx_right:.2f} pixels")
print(f"   ⚠️  ERROR: Focal lengths don't match!" if abs(fx_left - fx_right) > 50 else "   ✓ OK")

print("\n3. PRINCIPAL POINTS (image center)")
print(f"   Left:  cx = {cx_left:.1f}, cy = {K_L[1, 2]:.1f}")
print(f"   Right: cx = {cx_right:.1f}, cy = {K_R[1, 2]:.1f}")

print("\n4. EXPECTED DISPARITY VALUES")
print("   At different distances:")
for distance in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    disparity = (fx_left * baseline) / distance
    print(f"   {distance:.2f}m → disparity = {disparity:.1f} pixels")

print("\n5. TRANSLATION VECTOR (T)")
print(f"   T = {T.ravel()}")
print(f"   Baseline component (should be mostly in X): {T[0][0]:.6f} m")

print("\n6. ROTATION MATRIX (R)")
print(f"   Should be close to identity if cameras are parallel:")
print(f"{R}")
angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) * 180 / np.pi
print(f"   Rotation angle: {angle:.2f}°")
print(f"   ⚠️  ERROR: Cameras are rotated by {angle:.1f}°" if angle > 5 else "   ✓ OK")

print("\n7. DISTORTION COEFFICIENTS")
print(f"   Left:  {D_L.ravel()}")
print(f"   Right: {D_R.ravel()}")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

issues = []

if abs(baseline - 0.16) > 0.02:
    issues.append("❌ CRITICAL: Baseline is significantly wrong! Re-calibrate with correct physical distance.")

if angle > 5:
    issues.append("❌ Cameras have significant rotation. Ensure cameras are parallel during calibration.")

if abs(fx_left - fx_right) > 100:
    issues.append("⚠️  Focal lengths differ significantly. Check if both cameras have same settings.")

# Test depth calculation
print("\n8. TEST DEPTH CALCULATION")
print("   If you measure a flower at 0.40m and get disparity of 200 pixels:")
test_disparity = 200
test_depth = (fx_left * baseline) / test_disparity
print(f"   Calculated depth = {test_depth:.3f}m")
print(f"   Expected: 0.40m")
print(f"   Error: {abs(test_depth - 0.40)*100:.1f} cm")

if not issues:
    print("\n✓ Calibration looks reasonable.")
    print("\nIf depth is still wrong, the problem is likely:")
    print("  1. Disparity matching is failing (check rectified images)")
    print("  2. Images are too low resolution (increase SCALE_FOR_MATCHING)")
    print("  3. Poor texture/lighting for stereo matching")
else:
    print("\nISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")

print("\n" + "=" * 60)
