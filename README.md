# Biomimicry Robot: Automated Flower Pollination System

**Project Summary:** An autonomous robotic system for flower detection and pollination using stereo vision, YOLO-based object detection, and precision motor control. The robot navigates a workspace, detects flowers using deep learning, estimates 3D position via stereo depth, approaches targets with sub-centimeter accuracy, and performs pollination while tracking previously visited flowers to avoid re-pollination.

**Quick Checklist:** Python 3.11+ & CUDA → Install dependencies (`pip install ultralytics torch opencv-python adafruit-motorkit pynput`) → Calibrate cameras → Run `python ServerToLaptop/laptop_client_manual.py`

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Entry Points & Usage](#entry-points--usage)
3. [Environment & Configuration](#environment--configuration)
4. [Troubleshooting](#troubleshooting)
5. [Contact / Changelog](#contact)

---

## Quick Start

### Prerequisites
- **Hardware:** Raspberry Pi 5 (4GB+), 2× USB cameras (stereo pair), 2× Adafruit Motor HATs (I2C addresses 0x60, 0x61), stepper motors (rails, main, arm), vibration motor, Van de Graaff generator with GPIO control
- **Software:** Python 3.11+, CUDA 11.8+ (for GPU training/inference), OpenCV, Ultralytics YOLO
- **Network:** Laptop and Pi on same network (Tailscale recommended for reliability)

### Installation (Local Development)

```bash
# Clone repository
git clone https://github.com/averagepindakaasenjoyer/BiomimicryCode
cd BiomimicryCode

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics torch torchvision opencv-python numpy python-dotenv pynput

# For Raspberry Pi (additional hardware libraries)
pip install adafruit-circuitpython-motorkit gpiozero
```

### One-Line Demo
```bash

# On laptop 
python running pi_server.py

# On laptop
python ServerToLaptop/laptop_client_manual.py
# Type 'demo' to start autonomous pollination, or 'keyboard' for manual control
# Further explanation on startup 
```

### Step-by-Step First Run

1. **Calibrate stereo cameras** (one-time setup):
   ```bash
   python CalibrationSetup/capture_stereo_calibration.py  # Capture 20+ ChArUco board images
   python Depth/opencv_calibration3.py                     # Generate calibration file
   # Output: Cam_Params/stereo_charuco_calibration_16cm.npz
   ```

2. **Start Pi server** (on Raspberry Pi, make sure to activate venv):
   ```bash
   cd BiomimicryCode/ServerToLaptop
   python pi_server.py
   # Output: [Pi] Listening on 0.0.0.0:8000...
   ```

3. **Start laptop client** (on development laptop, activate venv):
   ```bash
   python ServerToLaptop/laptop_client_manual.py
   # Commands: demo (auto mode), keyboard (manual), reset, move, quit
   ```

4. **Train YOLO model** (optional, for improving detection):
   ```bash
   python YoloTrain.py  # Uses data.yaml, trains on Combined_Dataset2
   # Output: runs/detect/trainX/weights/best.pt
   cp runs/detect/trainX/weights/best.pt current_best_yolo.pt
   ```

## Entry Points & Usage

### 1. Robot Control System

#### Laptop Client (Main Entry Point)
**File:** `ServerToLaptop/laptop_client_manual.py`

**Usage:**
```bash
python laptop_client_manual.py
```

**Interactive Commands:**
- `demo` - Start autonomous flower detection and pollination demo
- `keyboard` - Enter keyboard control mode (WASD movement, O=arm up, P=pollinate)
- `reset` - Move all motors to home position (0,0,0)
- `move <motor> <steps>` - Manual motor control (e.g., `move rails 100`)
- `quit` - Exit program

**Configuration (edit in file):**
```python
PI_IP = "100.98.87.47"  # <<ASSUME: Check actual Pi IP with `tailscale status`>>
CONFIDENCE_THRESHOLD = 0.6  # Flower detection confidence threshold
USE_ADVANCED_FLOWER_ESTIMATION = False  # True = yellow circle detection, False = bbox center
POLLINATION_EXCLUSION_RADIUS_CM = 15.0  # No re-pollination within 15cm of visited flowers
DEBUG_MOVEMENT = True  # Pause and display movement calculations during approach
```

**Demo Mode Behavior:**
1. Square search pattern: (0,0) → (15,45) → (0,45) → (15,0) → (0,0) cm
2. Flower detection → Initial approach (move camera center to flower)
3. Refined approach with tighter vision crop (150px crop margin)
4. Final approach positioning arm tip on flower center
5. Pollination sequence: arm down → VDG 2s → arm up
6. Mark flower as pollinated, add to exclusion zone

**Keyboard Mode Controls:**
- `W/S` - Forward/backward (main axis)
- `A/D` - Left/right (rails axis)
- `O` - Arm up 1cm (single press)
- `P` - Pollinate sequence (single press)
- `V/B` - Van de Graaff ON/OFF
- `ESC` - Exit keyboard mode

#### Pi Server
**File:** `ServerToLaptop/pi_server.py`

**Usage (on Raspberry Pi):**
```bash
python pi_server.py
```

**Expected Output:**
```
[Pi] Initializing motor controllers...
[Pi] Opening cameras 0 and 2...
[Pi] Listening on 0.0.0.0:8000 for connections...
```

**Configuration:**
```python
CAM_LEFT = 0   # <<ASSUME: Check with `v4l2-ctl --list-devices`>>
CAM_RIGHT = 2
PORT = 8000
```

### 2. YOLO Training

#### Main Training Script
**File:** `YoloTrain.py`

**Usage:**
```bash
python YoloTrain.py
```

**Configuration (edit in file):**
```python
HYPERPARAM_TUNING['enable'] = True  # Run Ray Tune hyperparameter search before training
AUGMENTATION['enable'] = True  # Enable data augmentation
TRAINING_HYP['dropout'] = 0.2  # Dropout rate for regularization
DISABLE_CHUNKED_TRAINING = False  # Use chunked training for large datasets
```

**Output:**
- `runs/detect/trainX/weights/best.pt` - Best model checkpoint
- `runs/detect/trainX/results.csv` - Training metrics per epoch

#### Server Training (with auto-consolidation) Further explaination in HipsterYoloTrain\HIPSTER_DEPLOYMENT.md
**File:** `server_train.py`

```bash
python server_train.py --plots --report
# Auto-consolidates chunk results, generates summary report
```

#### HPC Cluster Training (HIPSTER)
```bash
# One-time setup
bash setup_env_hipster.sh

# Submit GPU job
sbatch yolo_train_gpu.sbatch

# Monitor
squeue -u $USER
tail -f /home/$USER/logs/yolo-gpu-train-<job_id>.out

# Check quota
quota -s
```

### 3. Inference & Testing

#### Single Image Inference
**File:** `YoloApply.py`

```bash
python YoloApply.py
# Edit image_path in main() to test on specific image
# Output: out_image.jpg with annotated detections
```

#### Live Depth Estimation
**File:** `Depth/flower_depth_live.py`

```bash
python Depth/flower_depth_live.py
# Live stereo feed with YOLO detection and depth overlay
# ESC to exit, SPACE to pause, 'r' to reset cameras
```

### 4. Calibration

#### Capture Calibration Images
```bash
python CalibrationSetup/capture_stereo_calibration.py
# SPACE to capture, ESC to quit
# Saves to CalibImg/ArucoImages16cm/
```

#### Run Calibration
```bash
python Depth/opencv_calibration3.py
# Output: Cam_Params/stereo_charuco_calibration_16cm.npz
```

#### Validate Calibration
```bash
python Depth/diagnose_stereo.py
# Shows baseline, focal length, expected disparities
```

### 5. Dataset Tools

```bash
# Check dataset integrity
python DatasetCreation/check_dataset_labels.py

# Combine multiple datasets
python DatasetCreation/CombineDatasets.py

# Split dataset
python DatasetCreation/TrainValSplit.py --train 0.7 --val 0.2 --test 0.1
```

---

## Environment & Configuration

### Required Environment Variables (`.env`)

```bash
# Dataset paths (relative to project root)
DATASET_DIR=../Combined_Dataset2
YAML_PATH=data.yaml

# Calibration files
PARAM_FOLDER=Cam_Params
STEREO_PARAMS_16CM=${PARAM_FOLDER}/stereo_charuco_calibration_16cm.npz
```

### Configuration Files

#### `data.yaml` (YOLO Dataset Config)
```yaml
train: ../Combined_Dataset2/images/train
val: ../Combined_Dataset2/images/val
test: ../Combined_Dataset2/images/test
nc: 1  # Number of classes
names:
  - flower
```

#### Network Configuration (in code)
```python
# laptop_client_manual.py
PI_IP = "100.98.87.47"  # <<ASSUME: Update with actual Pi IP>>
PORT = 8000

# Message protocol
MSG_TYPE_FRAME = 1    # Stereo frame data
MSG_TYPE_COMMAND = 2  # Motor commands
```

#### Motor Calibration
```python
# Physical parameters (laptop_client_manual.py)
WHEEL_DIAMETER_CM = 2.5
STEPS_PER_REV = 200  # NEMA 17 stepper motors
PIXELS_PER_CM = 25.84  # Calibrated with calibrate_pixels_per_cm.py

# Calibration factors (adjust for actual robot)
RAILS_CALIBRATION = 1.0  # <<ASSUME: Tune with movement tests>>
MAIN_CALIBRATION = 1.0
ARM_CALIBRATION_FACTOR = 1.0

# Arm tip offset from camera center
OFFSET_X_CM = 5.03   # Left offset
OFFSET_Y_CM = 5.81   # Forward offset
```

#### Position Limits
```python
LIMIT_X_MAX = 45.0  # Rails max (cm)
LIMIT_Y_MAX = 18.0  # Main max (cm)
LIMIT_Z_MAX = 20.0  # Arm max (cm)
```

### Required Ports
- **8000** - Pi server listening port (TCP)

### Data File Formats

#### Stereo Calibration (`*.npz`)
```python
# Keys: K_left, D_left, K_right, D_right, R, T
calib = np.load('Cam_Params/stereo_charuco_calibration_16cm.npz')
baseline = np.linalg.norm(calib['T'])  # Should be ~0.16m
```

#### YOLO Annotations (`.txt`)
```
# Format: class x_center y_center width height (normalized 0-1)
0 0.512 0.483 0.124 0.156
```

---

### A. Calibration Workflow

#### Step 1: Capture Calibration Images
```bash
python CalibrationSetup/capture_stereo_calibration.py
# 1. Position board at various angles/distances (20-30 images)
# 2. Press SPACE to capture each pose
# 3. Ensure good corner detection (board should fill 30-70% of frame)
# 4. ESC when done
```

#### Step 2: Run Calibration
```bash
python Depth/opencv_calibration3.py
# Expected output:
#   [INFO] Found 20 image pairs
#   [INFO] Valid stereo pairs: 20
#   [INFO] Stereo calibration RMS: 0.45 pixels (target: <0.5)
#   [INFO] Baseline: 0.1598 m (target: 0.16 m)
# Output: Cam_Params/stereo_charuco_calibration_16cm.npz
```

#### Step 3: Validate Calibration
```bash
python Depth/diagnose_stereo.py
# Check:
#   ✓ Baseline ~0.16m (±0.02m acceptable)
#   ✓ Focal lengths match (±50px acceptable)
#   ✓ Rotation angle <5°
# If calibration fails, recapture with better board positioning
```

#### Step 4: Calibrate Pixel-to-CM Conversion
```bash
python ServerToLaptop/calibrate_pixels_per_cm.py
# 1. Place ruler/object of known size in frame
# 2. Mark pixel coordinates of endpoints
# 3. Calculate PIXELS_PER_CM
# 4. Update value in laptop_client_manual.py
```

### B. Dataset Preparation (Optional - for retraining)

```bash
# 1. Organize images
mkdir -p Combined_Dataset2/{images,labels}/{train,val,test}

# 2. Label images (use LabelImg, Roboflow, or similar)
#    Export in YOLO format (class x_center y_center width height)

# 3. Validate dataset
python DatasetCreation/check_dataset_labels.py
# Expected: ~70% train, ~20% val, ~10% test

# 4. Update data.yaml with correct paths

# 5. Train model, should work on hipster cluster
python YoloTrain.py
# Monitor: tensorboard --logdir runs/detect/
```

### C. Running Full Demo

#### Startup Sequence

1. **Power on Pi, start server:**
   ```bash
   ssh pi@<tailscale-ip>
   cd BiomimicryCode/ServerToLaptop
   python pi_server.py
   # Wait for: [Pi] Listening on 0.0.0.0:8000...
   ```

2. **Start laptop client:**
   ```bash
   cd BiomimicryCode/ServerToLaptop
   python laptop_client_manual.py
   # Wait for: [Laptop] Connected to Pi...
   #           [Laptop] Frames initialized. Ready for commands.
   ```

3. **Run demo mode:**
   ```
   > demo
   [DEMO] Starting automatic flower tracking demo...
   [DEMO] Square search pattern: (0,0) → (15,45) → (0,45) → (15,0) → (0,0)
   ```

#### Expected Runtime Behavior

**Searching State:**
- Robot moves to next waypoint in search pattern
- Frame rate: ~3-5 FPS with YOLO detection
- When flower detected (confidence > 0.6):
  - Console: `[DEMO] Frame X: FLOWER DETECTED! Starting approach sequence`
  - Transitions to initial_approach state

**Initial Approach State:**
- Calculates offset from camera center to flower center
- Moves camera to align with flower (accounting for arm offset)
- If `DEBUG_MOVEMENT = True`: pauses and displays debug visualization
  - Shows flower bbox, camera center crosshair, arm tip marker
  - Shows pixel offsets, cm calculations, motor steps
  - Press 'q' in visualization window to continue
- After movement, transitions to refined_approach state

**Refined Approach State:**
- Crops vision to 150px margin around flower (tighter detection)
- Re-detects flower with tighter confidence (0.6)
- Calculates final positioning adjustments
- Filters out previously pollinated flowers (within 15cm exclusion zone)
- If flower still valid, transitions to pollinating state
- If flower is pollinated or lost, returns to searching

**Pollinating State:**
- Final check: flower not in exclusion zone
- Arm sequence:
  1. Move arm down (10s)
  2. Run Van de Graaff motor
  3. Move arm up (10s)
- Mark flower world position as pollinated
- Display: `[DEMO] [POLLINATE] Pollinated flowers count: N`
- Return to searching state

#### Keyboard Control Mode

```
> keyboard
[KEYBOARD] Entering keyboard control mode
[KEYBOARD] Controls:
  W: Forward, S: Backward, A: Left, D: Right
  O: Arm up 1cm, P: Pollinate sequence
  V: Van de Graaff ON, B: Van de Graaff OFF
  ESC: Exit
```

- Hold WASD keys for continuous movement (updates every 300ms)
- Camera display refreshes at ~60 FPS in separate thread
- Press O repeatedly to raise arm incrementally
- Press P to run full pollination sequence at current position

### D. Verifying Correct Results

#### Calibration Quality Checks
```bash
python Depth/diagnose_stereo.py
# ✓ Baseline: 0.160m (±0.01m = GOOD)
# ✓ Rotation angle: 2.3° (< 5° = GOOD)
# ✓ Focal length match: fx_L=2891, fx_R=2887 (< 50px diff = GOOD)
```

#### Detection Accuracy
```bash
python Depth/flower_depth_live.py
# Place flower at known distance (e.g., 40cm)
# Check displayed depth: "0.40m" (±5cm acceptable)
# Confidence should be > 0.6 for valid detections
```

#### Motor Movement Accuracy
```bash
# Test rails movement (X-axis)
> move rails 500
[Laptop] Sending motor command: {'rails': 500}
[Laptop] Current Position: X=6.39cm, Y=0.00cm, Z=0.00cm
# Measure actual movement with ruler, should be ~6.4cm
# If mismatch, adjust RAILS_CALIBRATION factor

# Test main movement (Y-axis)
> move main 500
# Measure, adjust MAIN_CALIBRATION if needed
```

#### Pollination Tracking
```bash
# After demo pollination:
> demo
[DEMO] [POLLINATE] Pollinated flowers count: 1
# Move to same flower location
[DEMO] All detected flowers have been pollinated, returning to search
# ✓ Flower correctly excluded from re-pollination
```

---

## Troubleshooting

### 1. Connection Issues

**Problem:** `[ERROR] Connection refused to 100.98.87.47:8000`

**Diagnosis:**
```bash
# On laptop
ping 100.98.87.47  # Check network connectivity
nc -zv 100.98.87.47 8000  # Test port connectivity
```

**Fixes:**
- Verify Pi server is running: `ssh pi@<ip> "ps aux | grep pi_server"`
- Check firewall: `sudo ufw status` on Pi
- Verify Tailscale: `tailscale status` on both devices
- Try direct IP if Tailscale fails: `ip addr show` on Pi

### 2. Camera Not Found

**Problem:** `[ERROR] Could not open camera 0` or `[ERROR] Could not open camera 2`

**Diagnosis:**
```bash
# On Pi
ls /dev/video*  # List available cameras
v4l2-ctl --list-devices  # Show camera names and indices
```

**Fixes:**
- Update `CAM_LEFT` and `CAM_RIGHT` indices in `pi_server.py` to match your setup
- Ensure cameras have power (check USB hub)
- Try different USB ports (some may be USB 2.0 only)
- Reboot Pi: `sudo reboot`

### 3. Poor Detection Accuracy

**Problem:** Flowers not detected or low confidence scores

**Diagnosis:**
```bash
python Depth/flower_depth_live.py  # Check live detection
# Look for confidence scores < 0.5
```

**Fixes:**
- Improve lighting (flowers need good illumination)
- Retrain model with more similar flower images
- Lower `CONFIDENCE_THRESHOLD` (but increases false positives)
- Check if camera is in focus
- Verify YOLO model loaded: check console for `[Laptop] Loading YOLO model from...`

### 4. Incorrect Depth Estimates

**Problem:** Depth shows 0.20m when flower is at 0.40m

**Diagnosis:**
```bash
python Depth/diagnose_stereo.py
# Check baseline and focal length values
```

**Fixes:**
- **Baseline wrong:** Re-run stereo calibration with correct camera spacing
- **Focal length mismatch:** Ensure both cameras have same resolution and settings
- **Disparity out of range:** Adjust `SCALE_FOR_MATCHING` (lower = larger depth range)
- **Poor calibration:** Recapture calibration images with better board coverage

### 5. Motor Movement Inaccurate

**Problem:** Command `move rails 500` moves 4cm instead of 6.4cm

**Diagnosis:**
```bash
# Test known distance
> move rails 1000
# Measure actual movement
actual_cm = <measured_value>
expected_cm = 1000 / STEPS_PER_CM  # Should be ~12.73cm
```

**Fixes:**
- Calculate correction: `RAILS_CALIBRATION = expected_cm / actual_cm`
- Update in `laptop_client_manual.py`
- Re-test movement
- Common causes: slipping wheels, loose belts, incorrect WHEEL_DIAMETER_CM

### 6. Robot Gets Stuck in Approach

**Problem:** Robot repeatedly approaches same flower without pollinating

**Diagnosis:**
- Check console for: `[DEMO] All detected flowers have been pollinated, returning to search`
- Indicates flower is in exclusion zone but logic error

**Fixes:**
- Verify `pollinated_flowers` list is populated: add debug print
- Check exclusion radius: may be too large (`POLLINATION_EXCLUSION_RADIUS_CM = 15.0`)
- Ensure world position calculation is correct (check `OFFSET_X_CM`, `OFFSET_Y_CM`)

### 7. YOLO Training Fails

**Problem:** `RuntimeError: CUDA out of memory`

**Diagnosis:**
```bash
nvidia-smi  # Check GPU memory usage
```

**Fixes:**
- Reduce batch size in `YoloTrain.py`: `batch=8` → `batch=4`
- Use smaller model: `yolo11n.pt` instead of `yolo11m.pt`
- Enable chunked training: `DISABLE_CHUNKED_TRAINING = False`
- Use CPU training: `device='cpu'` (much slower)

### 8. Hyperparameter Tuning Takes Forever

**Problem:** Ray Tune tuning runs for hours with no progress

**Diagnosis:**
```bash
# Check tuning config
grep -A 10 "HYPERPARAM_TUNING" YoloTrain.py
```

**Fixes:**
- Reduce iterations: `'iterations': 150` → `'iterations': 30`
- Reduce epochs per trial: `'epochs': 15` → `'epochs': 5`
- Disable tuning: `HYPERPARAM_TUNING['enable'] = False`
- Skip to main training: edit line to return immediately

### 9. Display Window Not Refreshing

**Problem:** Debug visualization frozen during `DEBUG_MOVEMENT = True`

**Diagnosis:**
- Check if OpenCV waitKey is called: should see "Press 'q' to continue..."

**Fixes:**
- Click on display window to give it focus
- Press 'q' key (not 'Q')
- If stuck, Ctrl+C and restart
- Disable debug: `DEBUG_MOVEMENT = False`

### 10. I2C Motor Communication Errors

**Problem:** `OSError: [Errno 121] Remote I/O error`

**Diagnosis:**
```bash
# On Pi
i2cdetect -y 1  # Should show 0x60, 0x61 with "60" and "61"
# If shows "UU", device is in use
# If shows "--", device not detected
```

**Fixes:**
- Enable I2C: `sudo raspi-config` → Interface Options → I2C → Enable
- Check wiring: ensure SDA/SCL connected correctly
- Verify I2C addresses: each Motor HAT has unique address (set via solder jumpers)
- Power cycle Pi and Motor HATs
- Check power supply: Motor HATs need 5-12V external power

### Logs Location
- **Laptop client:** Console output (no file logging by default)
- **Pi server:** Console output (no file logging by default)
- **YOLO training:** `training_logs/training_YYYYMMDD_HHMMSS.log`
- **SLURM jobs:** `/home/$USER/logs/yolo-gpu-train-<job_id>.out`

### Error Interpretation Examples

```python
# Good detection
[DEMO] Frame 45: Confidence=0.82, Depth=0.42m
# ✓ Confidence > 0.6, depth reasonable

# Poor detection
[DEMO] Frame 67: Confidence=0.32, Depth=0.42m
# Confidence too low, will be filtered out

# Depth error
[DEMO] Depth stats: median=inf, mean=inf, valid_pixels=0
# ✗ No valid depth (disparity too small or too large)
# FIX: Adjust SCALE_FOR_MATCHING or recalibrate

# Motor limit reached
[Laptop] Movement clamped: requested y=22.0cm, limited to 18.0cm
# Movement exceeded workspace limits (LIMIT_Y_MAX)
# Robot will move to boundary instead of requested position
```

### Known Limitations

1. **Flower Detection Range:** 0.25m - 50cm optimal. Below 0.25m, disparity exceeds range. Above 0.50m, resolution too low.
2. **Movement Speed:** ~1cm/s (limited by stepper motor speed and frame rate). Full approach takes 5-10s.
3. **Workspace Size:** 45cm × 18cm × 20cm (rails × main × arm). Hardcoded in position limits.
4. **Lighting Sensitivity:** YOLO performance drops in dim light. Requires 300+ lux for reliable detection.
5. **Flower Similarity:** Model trained on specific flower types. Struggles with very different morphology.
6. **Frame Rate:** 3-5 FPS during detection (YOLO inference bottleneck). Faster GPU improves performance.
7. **Network Dependency:** Cannot run offline (Pi-Laptop communication required). Consider edge deployment for autonomy.

### Important Pitfalls

1. **I2C Address Conflicts:** Motor HATs must have unique addresses (0x60, 0x61). Default is 0x60; solder jumpers to change.
2. **Camera Rotation:** USB cameras may mount upside-down. `SWAP_CAMERAS` and `ROTATE_LEFT/RIGHT` handle 180° rotations. Adjust if images inverted.
3. **Focal Length Scaling:** When downscaling images for stereo, must scale intrinsic matrix (`K_L *= scale`). Missing this breaks depth calculation.
4. **Motor Release:** Always release motors after movement (`motor.release()`) to prevent overheating and power drain.
5. **Position Tracking:** Position is tracked in software, not with encoders. Any motor slip accumulates error. Reset position periodically.
6. **Exclusion Zone Coordinate System:** Pollinated flowers stored in world coordinates, not camera coordinates. Requires accurate position tracking.
7. **Vibration Motor Timing:** Continuous motors (vibration, VDG) use time-based control, not step-based. Ensure duration matches physical response.
8. **ChArUco Board Quality:** Print on stiff material, ensure flat. Warped boards cause calibration errors.
9. **Dataset Imbalance:** Include "no flower" images in training to reduce false positives. ~30% negative samples recommended.
10. **GPU Memory Leaks:** YOLO model loading can leak memory over time. Restart client after ~100 detections if performance degrades.

### Contact

**Original Authors:** 
- **Conributer 1:** [Matthias Meijer] - [averagepindakaasenjoyer]
- **Conributer 2:** [Thijn van Veen] - [Kapithijn]


**Last Updated:** 2026-01-29
