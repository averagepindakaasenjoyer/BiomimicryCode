# Computation Timing Feature

## Overview
The laptop client now includes computation time tracking to measure performance per frame.

## Usage

### 1. Enable Timing
In [laptop_client.py](laptop_client.py), set the flag to `True`:
```python
RECORD_TIMING = True
```

### 2. Run the Client
Execute the laptop client as normal. It will:
- Create `computation_timing.csv` in the ServerToLaptop directory
- Log timing data for each processed frame

### 3. CSV Output Format
The CSV contains:
- **timestamp**: Date and time of frame processing
- **frame_id**: Sequential frame number
- **detection_time_ms**: YOLO flower detection time (milliseconds)
- **depth_computation_time_ms**: Stereo disparity + depth map computation (milliseconds)
- **total_processing_time_ms**: Total processing time (depth + detection + ROI depth estimation)
- **num_detections**: Number of flowers detected in frame

### 4. Analyze Results
Use Python/pandas to compute statistics:
```python
import pandas as pd

df = pd.read_csv('computation_timing.csv')

# Average times
print(f"Avg Detection Time: {df['detection_time_ms'].mean():.2f} ms")
print(f"Avg Depth Time: {df['depth_computation_time_ms'].mean():.2f} ms")
print(f"Avg Total Time: {df['total_processing_time_ms'].mean():.2f} ms")

# Frames per second
avg_total_ms = df['total_processing_time_ms'].mean()
fps = 1000 / avg_total_ms
print(f"Effective FPS: {fps:.2f}")
```

## Performance Metrics Tracked

### Detection Time
- YOLO model inference on full-resolution left frame
- Includes preprocessing and postprocessing

### Depth Computation Time
- SGBM stereo matching on rectified frames
- Disparity to depth map conversion (using Q matrix)

### Total Processing Time
- Depth computation
- YOLO detection
- ROI depth estimation for all detections
- Target selection logic

### Excluded from Timing
- Frame reception over network
- Stereo rectification (remapping)
- Motor command calculation
- Display rendering

## Notes
- Timing has minimal overhead (~3 microseconds per time.time() call)
- CSV is appended in real-time (no buffering)
- Set `RECORD_TIMING = False` when not profiling to avoid disk I/O
