# Flower Depth Estimation - Live Script

## Overview
`flower_depth_live.py` performs **real-time stereo depth estimation for detected flowers**.

**Key features:**
- Live dual-camera capture (cameras 1 & 2)
- YOLO detection with confidence threshold (0.8 by default)
- Per-flower depth statistics (median, mean, min/max, coverage %)
- Adjustable frame rate, disparity search range, and display options
- Real-time FPS counter and detection overlay

## Usage

```bash
python DepthByRico/flower_depth_live.py
```

## Controls

| Key    | Action                      |
|--------|---------------------------|
| **ESC** | Exit                       |
| **SPACE** | Pause/Resume capture      |
| **F**  | Toggle fullscreen         |
| **R**  | Reset stereo rectification |

## Configuration

Edit the script's top-level config section to adjust:

- **`YOLO_MODEL_PATH`** = Path to YOLO model (default: `current_best_yolo.pt`)
- **`CONFIDENCE_THRESHOLD`** = Min YOLO confidence to accept detection (default: 0.8)
- **`TARGET_FPS`** = Frame processing rate (default: 10)
- **`SCALE_FOR_MATCHING`** = Downscale factor for stereo matching (default: 0.5; use 0.4–0.45 if disparities clip)
- **`EXPECTED_DISTANCE`** = Target depth for disparity range tuning (default: 0.40 m)
- **`MIN_DEPTH`** / **`MAX_DEPTH`** = Valid depth range for visualization
- **`SHOW_DISPARITY`** = Overlay disparity heatmap in top-left corner

## Output

On each detected flower, displays:
- **Bounding box** (green if depth is valid, red if invalid)
- **Confidence** from YOLO
- **Median depth** (meters)
- **Coverage %** (valid pixels in ROI / total ROI pixels)

### Terminal Output
Frame-by-frame console shows:
- Current FPS
- Number of detections
- (Optional) Disparity & depth map stats

## Calibration

Uses stereo calibration from:
```
Cam_Params/stereo_charuco_calibration_16cm_rico.npz
```

If depth estimates seem wrong:
1. Verify cameras 1 & 2 match the calibration rig
2. Try adjusting `SCALE_FOR_MATCHING` (lower → shorter range, higher disparities)
3. Increase `YOLO_CONF` if too many false positives affect matching
4. Ensure good lighting and distinct texture in scene

## Troubleshooting

### "Could not open cameras"
- Check camera indices (`CAM_LEFT`, `CAM_RIGHT`)
- Run `capture_cameras.py` to verify camera connectivity

### Low depth validity (< 20%)
- Lower `SCALE_FOR_MATCHING` to 0.4 or 0.35
- Improve scene texture/lighting
- Verify rectification quality (check `stereo_output/rectified_left.png`)

### Depth estimates too large/small
- Check baseline in calibration (should be ~0.16 m)
- Try `SWAP_CAMERAS = True` (temporarily) in `disparity.py` to test camera order
- Re-calibrate if rig has changed

## See Also
- [disparity.py](disparity.py) – Standalone depth estimation for images
- [capture_cameras.py](../capture_cameras.py) – Manual stereo frame capture
