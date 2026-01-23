# Stereo Vision Robot Control System

Real-time stereo vision-based flower tracking and robotic arm control system. The Pi captures stereo camera frames and sends them to a laptop for YOLO-based flower detection and stereo depth estimation. The laptop computes motor commands to track and approach flowers, which are executed on the Pi.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Raspberry Pi (pi_server.py)                │
│  ┌─────────────┐         ┌──────────────────┐           │
│  │ Camera Left │  <---→  │ Camera Right     │           │
│  │ (CAM_LEFT)  │         │ (CAM_RIGHT)      │           │
│  └─────────────┘         └──────────────────┘           │
│         ↓                                                 │
│    Capture @ 1920x1080, 30fps                            │
│         ↓                                                 │
│  ┌─────────────────────────────────────────┐            │
│  │  Stereo Frame Serialization (JPEG)      │            │
│  └─────────────────────────────────────────┘            │
│         ↓ [Network: TCP port 8000]                       │
└─────────────────────────────────────────────────────────┘
         ↓                                         ↑
      SEND FRAMES                           RECV COMMANDS
         ↓                                         ↑
┌─────────────────────────────────────────────────────────┐
│            Laptop (laptop_client.py)                    │
│  ┌──────────────────────────────────────────┐           │
│  │  YOLO Flower Detection                   │           │
│  │  + Stereo Depth Estimation               │           │
│  │  + Motor Command Generation              │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
         ↓
   ┌─────────────────┐
   │  CV2 Display    │
   │  - Left Frame   │
   │  - Right Frame  │
   │  (side-by-side) │
   └─────────────────┘
```

## Files Overview

### `pi_server.py`
Runs on Raspberry Pi. Captures stereo frames and handles motor commands.

**Key Functions:**
- `send_stereo_frames(conn, frame_left, frame_right)` - Encode and transmit camera frames
- `receive_motor_command(conn)` - Listen for motor instructions
- `execute_motor_command(command_dict)` - Drive motors based on received commands
- `motor_step_worker(motor_obj, steps, ...)` - Low-level motor stepping

**Dependencies:**
- OpenCV (`cv2`)
- Adafruit Motor Kit (`adafruit_motor`, `adafruit_motorkit`)
- Board (`board`)

**Configuration:**
```python
PORT = 8000                    # Network port
CAM_LEFT = 1                   # Left camera device index
CAM_RIGHT = 2                  # Right camera device index
STEP_DELAY = 0.01              # Stepper motor stepping interval (seconds)
STEP_DELAY_ARM = 0.01          # Arm motor stepping interval (seconds)
```

**Motor Setup:**
- `kit1` (address 0x60): Controls `rails` (stepper1) and `main` (stepper2)
- `kit2` (address 0x61): Controls `arm` (stepper1)

---

### `laptop_client.py`
Runs on the laptop. Performs vision processing and generates motor commands.

**Key Functions:**
- `detect_flowers(frame, conf_threshold)` - YOLO detection
- `compute_stereo_disparity(rectL, rectR, K_L_use)` - Stereo matching
- `compute_depth_map(disparity, Q)` - Disparity to depth conversion
- `estimate_roi_depth(depth_map, roi_box, ...)` - ROI depth statistics
- `select_target_flower(detections, depth_stats_list, ...)` - Target selection logic
- `convert_offsets_to_motor_steps(dx_pixels, dy_pixels)` - Pixel offset → motor steps
- `convert_depth_to_arm_steps(depth_m, target_depth_m)` - Depth error → arm steps

**Dependencies:**
- OpenCV (`cv2`)
- PyTorch (`torch`)
- Ultralytics YOLO (`ultralytics`)
- NumPy (`numpy`)

**Configuration:**
```python
# Network
PI_IP = "100.98.87.47"                    # Pi's IP address
PORT = 8000

# Vision
YOLO_MODEL_PATH = "../current_best_yolo.pt"
CALIB_PATH = '../Cam_Params/stereo_charuco_calibration_16cm.npz'
CONFIDENCE_THRESHOLD = 0.6                # Minimum YOLO confidence
YOLO_CONF = 0.5
SCALE_FOR_MATCHING = 0.5                  # Downscale factor for stereo matching

# Depth
MIN_DEPTH = 0.25                          # Minimum valid depth (meters)
MAX_DEPTH = 2.0                           # Maximum valid depth (meters)
EXPECTED_DISTANCE = 0.40                  # Target working distance (meters)

# Display
SHOW_DEBUG = True                         # Show live video windows
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Timing
RECORD_TIMING = True                      # Log computation times to CSV
TIMING_CSV_PATH = "computation_timing.csv"

# Motor Parameters
WHEEL_DIAMETER_CM = 2.5
PIXELS_PER_CM = 10.0
MAX_CM_PER_CYCLE = 10.0
```

---

## Setup Instructions

### Prerequisites
- Raspberry Pi with stereo cameras configured as `/dev/video1` and `/dev/video2`
- Laptop with Python 3.8+ and CUDA support (optional but recommended)
- Calibration file: `stereo_charuco_calibration_16cm.npz` in `Cam_Params/`
- YOLO model: `current_best_yolo.pt` in root directory

### Pi Setup

1. **Install dependencies:**
   ```bash
   pip install opencv-python adafruit-motor adafruit-motorkit adafruit-blinka
   ```

2. **Configure I2C motor controllers:**
   - Address `0x60`: Controls rails and main motors
   - Address `0x61`: Controls arm motor
   - Verify with: `i2cdetect -y 1`

3. **Identify cameras:**
   ```bash
   ls -l /dev/video*
   ```
   Update `CAM_LEFT` and `CAM_RIGHT` in `pi_server.py` if needed.

4. **Run server:**
   ```bash
   python pi_server.py
   ```
   Will wait for laptop connection before starting camera capture.

### Laptop Setup

1. **Install dependencies:**
   ```bash
   pip install opencv-python torch torchvision ultralytics numpy
   ```
   For GPU: Install CUDA-enabled PyTorch from https://pytorch.org

2. **Update configuration:**
   - Set `PI_IP` to your Raspberry Pi's IP address
   - Verify paths for YOLO model and calibration files

3. **Run client:**
   ```bash
   python laptop_client.py
   ```
   Will attempt to connect to Pi and begin processing stereo frames.

---

## Operation

### Starting the System

1. **On Raspberry Pi:**
   ```bash
   cd ServerToLaptop
   python pi_server.py
   # Output: "[Pi] Waiting for connection on port 8000..."
   ```

2. **On Laptop (in a separate terminal):**
   ```bash
   cd ServerToLaptop
   python laptop_client.py
   ```
   
3. **Verify connection:**
   - Pi should print: `[Pi] Connected to <laptop_ip>`
   - Laptop should print: `[Laptop] Connected to Pi at <pi_ip>:8000`
   - Live video window appears showing both cameras side-by-side

### Display Windows

The laptop shows a side-by-side view:
- **Left side**: Left camera with YOLO detections and tracking overlays
  - Green boxes = target flower
  - Gray boxes = other detections
  - Blue crosshair = frame center
  - Yellow line = offset from center to target
- **Right side**: Right camera (reference, no annotations)

### Data Recording

If `RECORD_TIMING = True`, computation times are logged to `computation_timing.csv`:
```csv
timestamp,frame_id,detection_time_ms,depth_computation_time_ms,total_processing_time_ms,num_detections
```

### Stopping the System

- Press **Q** or **ESC** in the video window
- Or send **Ctrl+C** to either process

---

## Motor Control

### Command Dictionary Format
```python
motor_command = {
    "rails": 100,      # Left/right movement (steps)
    "main": -50,       # Forward/backward movement (steps)
    "arm": 20          # Up/down arm movement (steps)
}
```

### Movement Strategy

1. **Horizontal Centering** (rails motor):
   - If flower left of center: move left
   - If flower right of center: move right
   - Threshold: ±30 pixels

2. **Vertical Centering** (main motor):
   - If flower above center: move forward
   - If flower below center: move backward
   - Threshold: ±30 pixels

3. **Depth Control** (arm motor):
   - If well-centered (±30 pixels) and depth ≠ target:
   - Move arm up if too far
   - Move arm down if too close
   - Target distance: 0.40 m (configurable)

4. **Search Behavior**:
   - If no valid flower detected: slow backward movement to search

---

## Troubleshooting

### "Connection refused"
- Verify Pi is running: `python pi_server.py`
- Check IP address with: `hostname -I` on Pi
- Ensure both devices are on same network
- Disable firewall or allow port 8000

### "Failed to read frames" or "Failed to receive frames"
- Verify cameras are connected to Pi
- Test with: `v4l2-ctl --list-devices`
- Check camera indices match `CAM_LEFT`, `CAM_RIGHT`
- Look for USB camera permission issues: add user to `video` group

### No YOLO detections
- Verify model path in `YOLO_MODEL_PATH`
- Check model file exists and is not corrupted
- Adjust `YOLO_CONF` (lower = more detections, more false positives)
- Adjust `CONFIDENCE_THRESHOLD` (post-processing filter)

### Stereo disparity computation slow/failing
- Check `SCALE_FOR_MATCHING` (lower = faster but less precise)
- Adjust `numDisparities` parameters in `compute_stereo_disparity()`
- Ensure calibration file is valid: `stereo_charuco_calibration_16cm.npz`

### Motors not moving
- Test motors directly with: `python -c "from adafruit_motorkit import MotorKit; kit = MotorKit(address=0x60); kit.stepper1.onestep()"`
- Verify I2C addresses: `i2cdetect -y 1`
- Check power supply to motor controllers
- Review motor log output on Pi

### Timing measurements show high latency
- Reduce `SCALE_FOR_MATCHING` for faster processing
- Lower `YOLO_CONF` threshold (faster inference but less accurate)
- Close other applications on laptop
- Use GPU: set `device = "cuda:0"` in `laptop_client.py`

---

## Performance Metrics

### Expected Performance (with GPU)
- YOLO detection: ~30-50 ms
- Stereo depth: ~100-150 ms
- Total cycle time: ~150-200 ms (~5-7 FPS)

### Expected Performance (CPU-only)
- YOLO detection: ~200-300 ms
- Stereo depth: ~200-300 ms
- Total cycle time: ~400-600 ms (~1.5-2.5 FPS)

---

## Advanced Configuration

### Adjusting Stereo Parameters

In `compute_stereo_disparity()`:
- `blockSize`: Increase for smoother but less detailed depth (3 or 5)
- `numDisparities`: Range of disparity search (16-256, multiple of 16)
- `uniquenessRatio`: Higher = stricter matching (6-10)
- `speckleWindowSize`: Remove small noise regions (larger = more filtering)

### Adjusting Motor Responsiveness

- `PIXELS_PER_CM`: Higher = smaller movements per command
- `MAX_CM_PER_CYCLE`: Limit maximum movement per frame
- `scale_move`: Multiply all movements by this factor (0.1 = 10% scale)
- `STEP_DELAY`: Motor stepping speed (lower = faster)

### Adjusting Target Selection

In `select_target_flower()`:
- Change weighting: `score = depth * 0.7 + dist_from_center * 0.0003`
- Prioritize closer flowers: increase `depth` coefficient
- Prioritize centered flowers: increase `dist_from_center` coefficient

---

## File Structure

```
ServerToLaptop/
├── laptop_client.py              # Main laptop processing
├── pi_server.py                  # Main Pi server
├── README.md                      # This file
├── TIMING_README.md               # Timing documentation
├── computation_timing.csv         # Performance logs (generated)
└── [laptop_client output]         # Any generated data files
```

---

## License & Attribution

Biomimicry robotics project - Flower tracking system with stereo vision and YOLO detection.

