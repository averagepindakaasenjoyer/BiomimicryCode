# CombinedMove.py
# Combines smooth multithreaded motor control with image-based movement.
# Based on original MoveImage.py and MultiThreadMove.py (combined & refactored).
#
# Requirements:
#  - adafruit_motorkit, adafruit_motor.stepper, board
#  - numpy, opencv-python
#  - A USB camera available as /dev/video0 (cv2.VideoCapture(0))
#
# How it works (short rails):
#  - detect_circle() finds the largest yellow-ish circular blob and returns its normalized center.
#  - process_image() returns pixel offsets from image center.
#  - offsets are mapped to desired linear movements (cm) by a simple proportional mapping (pixels / PIXELS_PER_CM).
#  - movement plan is converted into per-motor step counts (using STEPS_PER_CM).
#  - each motor is driven in its own thread with a per-step delay (STEP_DELAY) for smooth motion.
#  - main loop processes one image, waits for movement to finish, then repeats.

# How it works (arm):
# - find flower using Yolo model
# - calcualte depth position based on stereo vision
# - move arm up/down based on depth position




import time
import threading
import board
import numpy as np
import cv2
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit
import sys
import signal

# ---------- Hardware setup ----------
kit1 = MotorKit(i2c=board.I2C(), address=0x60)
kit2 = MotorKit(i2c=board.I2C(), address=0x61)

motor_dict = {
    "rails" : kit1.stepper1,
    "main": kit1.stepper2,
    "arm": kit2.stepper1,
}


# Motor logical directions (how we treat a "front"/"rear"/"left"/"right" move)
# Each tuple: (motor_name, direction_multiplier)
direction_dict = {
    "front": [("main", 1), ("main", 1)], #moving main rails forward
    "rear": [("main", -1), ("main", -1)], #moving main rails backward
    "right": [("rails", 1), ("rails", 1)], #moving top rails to the right
    "left": [("rails",-1), ("rails", -1)], # moving top rails to the left
    "up": [("arm", 1)], # moving arm up
    "down": [("arm", -1)], # moving arm down
}

# ---------- Motion parameters rails and Big rails ----------
WHEEL_DIAMETER_CM = 2.5
CIRCUMFERENCE_CM = WHEEL_DIAMETER_CM * np.pi
STEPS_PER_REV = 200                # Nema17 typical
STEPS_PER_CM = STEPS_PER_REV / CIRCUMFERENCE_CM
scale_move = 0.1 # Scale factor for movement responsiveness
# How many image pixels correspond to 1 cm of robot motion (approx).
PIXELS_PER_CM = 10.0

# Step delay for smooth motion (seconds). Lower =faster. Keep >= ~0.003 for most steppers.
STEP_DELAY = 0.01

# Maximum allowed movement per cycle (safety)
MAX_CM_PER_CYCLE = 10.0
MAX_STEPS_PER_CYCLE = int(MAX_CM_PER_CYCLE * STEPS_PER_CM)


# ---------- Arm parameters ----------
WHEEL_DIAMETER_CM_ARM = 4.3 # in cm
CIRCUMFERENCE_CM_ARM = WHEEL_DIAMETER_CM_ARM * np.pi
STEPS_PER_REV_ARM = 200  # Steps per full revolution of the stepper motor
STEPS_PER_CM_ARM = STEPS_PER_REV_ARM / CIRCUMFERENCE_CM_ARM
STEP_DELAY_ARM = 0.01

# Class saving the current position of entire arm and found flowers

class RobotPose:
    def __init__(self, x_limit_cm= 47, y_limit_cm= 15.7, z_limit_cm=45.0):
        # All units in cm
        self.x = 0.0      # left / right (rails)
        self.y = 0.0      # front / rear (main)
        self.z = 0.0      # arm height
        self.pollinated_flowers = []  # list of (x, y, z) positions of flowers
        self.x_limit_cm = x_limit_cm
        self.y_limit_cm = y_limit_cm
        self.z_limit_cm = z_limit_cm


        self.lock = threading.Lock()
    
    def print_pose(self):
        with self.lock:
            print(f"Robot Pose - X: {self.x:.2f} cm, Y: {self.y:.2f} cm, Z: {self.z:.2f} cm")

    def update_xy(self, dx_cm, dy_cm):
        with self.lock:
            self.x += dx_cm
            self.y += dy_cm

    def update_z(self, dz_cm):
        with self.lock:
            self.z += dz_cm

    def reset(self):
        with self.lock:
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    def snapshot(self):
        with self.lock:
            return (self.x, self.y, self.z)
    
    def add_flower(self, x, y, z):
        with self.lock:
            self.pollinated_flowers.append((x, y, z))
    
        




# Hough detection tuning
MIN_YELLOW_HSV = np.array([20, 100, 100])
MAX_YELLOW_HSV = np.array([30, 255, 255])

# ---------- Utility functions ----------
def clamp(n, small, large):
    return max(small, min(n, large))

# Motor stepping worker: moves a single motor a given number of steps (signed).
def motor_step_worker(motor_obj, steps, step_delay=STEP_DELAY, style=stepper.DOUBLE):
    """
    Drive 'motor_obj' for 'steps' steps. Steps is signed: positive -> FORWARD, negative -> BACKWARD.
    This runs in its own thread.
    """
    if steps == 0:
        return
    direction = stepper.FORWARD if steps > 0 else stepper.BACKWARD
    steps = abs(int(steps))
    try:
        for _ in range(steps):
            motor_obj.onestep(direction=direction, style=style)
            time.sleep(step_delay)
    except Exception as e:
        # Best-effort: stop gracefully on error
        print(f"[motor error] {e}")
    finally:
        try:
            motor_obj.release()
        except Exception:
            pass

# Release all motors (call when exiting)
def release_all_motors():
    for m in motor_dict.values():
        try:
            m.release()
        except Exception:
            pass

# ---------- Image processing ----------
def detect_circle(image):
    """
    Detect yellow circle-like blobs and return center normalized (x_pct, y_pct).
    Returns None if nothing found.
    """
    if image is None:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, MIN_YELLOW_HSV, MAX_YELLOW_HSV)
    masked = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.medianBlur(gray, 7)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    h, w = image.shape[:2]
    min_r = max(10, int(min(h, w) * 0.03))
    max_r = int(min(h, w) * 0.45)

    # HoughCircles parameters tuned for robust detection of a single prominent circle
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=max(h, w) // 2,
        param1=100,
        param2=40,
        minRadius=min_r,
        maxRadius=max_r
    )

    if circles is None:
        return None

    circles = np.uint16(np.around(circles))
    # take the first (largest/only) detection
    cx, cy, r = circles[0, 0]
    return (cx / w, cy / h)

def process_image_once(cap):
    """
    Capture a frame from given cv2.VideoCapture and return pixel offsets (dx, dy) from center.
    dx positive => circle is to the right; dy positive => circle is down.
    Returns None on failure.
    """
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    detected = detect_circle(frame)
    if detected is None:
        return (None, None, frame)  # no circle: zero offsets
    cx_norm, cy_norm = detected
    h, w = frame.shape[:2]
    cx_px = int(cx_norm * w)
    cy_px = int(cy_norm * h)
    dx = cx_px - (w // 2)
    dy = cy_px - (h // 2)
    return (dx, dy, frame)

# ---------- Motion planning ----------
def convert_offsets_to_motor_steps(dx_pixels, dy_pixels):
    """
    Convert pixel offsets to per-motor step counts.
      - dx controls main forward/rear movement
      - dy controls left/right rail movement
    Returns dict: { motor_name: signed_step_count, ... }
    """
    # Map pixels -> cm using PIXELS_PER_CM
    dx_cm = dx_pixels / PIXELS_PER_CM
    dy_cm = dy_pixels / PIXELS_PER_CM

    # clamp per-cycle
    dx_cm = clamp(dx_cm, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE)
    dy_cm = clamp(dy_cm, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE)

    # Determine which logical movement we need and magnitude (we'll split into per-motor later)
    # Positive dx_cm means object is to the right -> we probably need to move "front" or "rear".
    # In original code they used X/10 -> we preserve sign and magnitude logic.
    # We'll map dx_cm > 0 -> move front (positive forward), dx_cm < 0 -> move rear.
    move_plan = {}  # motor_name -> signed steps

    # main (X axis)
    if abs(dx_cm) >= 0.5:  # deadzone ~ 0.5 cm
        # choose 'front' for positive dx_cm (object is right) - this depends on camera/robot alignment
        # Use the direction_dict mapping to split between rear_main and front_main
        if dx_cm > 0:
            entries = direction_dict["left"]
        else:
            entries = direction_dict["right"]
        # For each motor entry: multiplier is +/-1, we compute steps proportional to abs(dx_cm)
        steps_for_cm = abs(dx_cm) * STEPS_PER_CM * scale_move
        for motor_name, multiplier in entries:
            # the original code used per-motor multipliers (some were +/-); keep sign in multiplier
            motor_steps = int(multiplier * steps_for_cm)
            # sum into plan
            move_plan[motor_name] = move_plan.get(motor_name, 0) + motor_steps

    # rails (Y axis) - typically slide left/right
    if abs(dy_cm) >= 0.5:
        if dy_cm > 0:
            entries = direction_dict["rear"]
        else:
            entries = direction_dict["front"]
        steps_for_cm = abs(dy_cm) * STEPS_PER_CM * scale_move
        for motor_name, multiplier in entries:
            motor_steps = int(multiplier * steps_for_cm)
            move_plan[motor_name] = move_plan.get(motor_name, 0) + motor_steps

    # Safety cap: ensure no motor exceeds max steps per cycle
    for k in list(move_plan.keys()):
        capped = clamp(move_plan[k], -MAX_STEPS_PER_CYCLE, MAX_STEPS_PER_CYCLE)
        move_plan[k] = int(capped)

    return move_plan


def convert_depth_to_arm_steps(depth_cm, current_arm_pos_cm= 0.0):
    """
    Convert depth position to arm motor steps.
    depth_cm: desired depth position in cm
    current_arm_pos_cm: current arm position in cm assume 0.0 since we have no feedback and move it back to 0 each time
    Returns signed step count for arm motor.
    """
    # Simple proportional control: move arm to reach desired depth
    error_cm = depth_cm - current_arm_pos_cm
    # Clamp movement to reasonable range
    error_cm = clamp(error_cm, -MAX_CM_PER_CYCLE, MAX_CM_PER_CYCLE)
    steps = int(error_cm * STEPS_PER_CM_ARM)
    return steps

def debug_window(frame, dx, dy):
            if dx is None or dy is None:
                dx, dy = 0, 0
            h, w = frame.shape[:2]
            cx = w // 2 + int(dx)
            cy = h // 2 + int(dy)
            cv2.circle(frame, (cx, cy), 10, (0,255,0), -1)
            cv2.line(frame, (w//2, h//2), (cx, cy), (255,0,0), 2)
            cv2.imshow("frame", frame)
            
# ---------- Main loop ----------
def main_loop(camera_index=0, show_debug=False):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: cannot open camera index", camera_index)
        return

    robot_pose = RobotPose()  # Initialize pose tracking

    try:
        while True:
            robot_pose.print_pose()
            # Capture and process image
            proc = process_image_once(cap)
            if proc is None:
                print("Camera read failed, retrying...")
                time.sleep(0.2)
                continue
            dx, dy, frame = proc
            # Print debug info
            print(f"dx(pixels)={dx}, dy(pixels)={dy}")
            if dx is None or dy is None:
                print("No circle detected → creeping rear")

                # slow, safe backward movement
                slow_steps = int(0.5 * STEPS_PER_CM)  # ~0.5 cm per cycle

                motor_obj = motor_dict.get("main")
                if motor_obj is not None:
                    motor_step_worker(
                        motor_obj,
                        slow_steps,          # positive = rear
                        step_delay=STEP_DELAY * 2  # slower than normal
                    )

                time.sleep(0.2)
                if show_debug:
                    debug_window(frame, dx, dy)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            
            # Plan movement
            plan = convert_offsets_to_motor_steps(dx, dy)
            if not plan:
                # nothing to move
                if show_debug:
                    cv2.putText(frame, "No target", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                time.sleep(0.08)
                continue

            # Start motor threads concurrently
            threads = []
            for motor_name, steps in plan.items():
                motor_obj = motor_dict.get(motor_name)
                if motor_obj is None:
                    print("Unknown motor:", motor_name)
                    continue
                if steps == 0:
                    continue
                t = threading.Thread(target=motor_step_worker, args=(motor_obj, steps, STEP_DELAY))
                t.daemon = True
                threads.append(t)
                t.start()

            # Optional visual debug overlay
            if show_debug:
                debug_window(frame, dx, dy)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Wait for movement to finish before processing next image (simple, robust)
            for t in threads:
                t.join()

            # Update robot pose based on movement (with limit checking)
            dx_cm = (dx / PIXELS_PER_CM) * scale_move if dx is not None else 0
            dy_cm = (dy / PIXELS_PER_CM) * scale_move if dy is not None else 0
            
            # Check limits before updating
            current_x, current_y, current_z = robot_pose.snapshot()
            new_x = current_x + dx_cm
            new_y = current_y + dy_cm
            
            if abs(new_x) <= robot_pose.x_limit_cm and abs(new_y) <= robot_pose.y_limit_cm:
                robot_pose.update_xy(dx_cm, dy_cm)
            else:
                print(f"Movement blocked: would exceed limits. Current: ({current_x:.2f}, {current_y:.2f})")

            # short delay to avoid spamming camera & motors
            time.sleep(0.05)

            # if the arm is above the flower based on the earlier calculations, move the arm up/down
            # For simplicity, we assume a fixed desired depth for now (e.g., 15 cm)
            if abs(dx) < 20 and abs(dy) < 20:
                desired_depth_cm = 15.0
                """
                add depth calcualtions HERE !!!!!! to replace desired_depth_cm variable
                
                """

                arm_steps = convert_depth_to_arm_steps(desired_depth_cm)
                if arm_steps != 0:
                    motor_obj = motor_dict.get("arm")
                    
                    # Check arm limits before moving
                    _, _, current_z = robot_pose.snapshot()
                    new_z = current_z + desired_depth_cm
                    
                    if 0 <= new_z <= 45.0:
                        motor_step_worker(motor_obj, arm_steps, STEP_DELAY_ARM)
                        robot_pose.update_z(desired_depth_cm)  # Move arm down
                        
                        # Assume flower pollinated at current position
                        x, y, z = robot_pose.snapshot()
                        robot_pose.add_flower(x, y, z)
                        print(f"Flower pollinated at position: ({x:.2f}, {y:.2f}, {z:.2f})")
                        
                        motor_step_worker(motor_obj, -arm_steps, STEP_DELAY_ARM)  # move back to original position
                        robot_pose.update_z(-desired_depth_cm)  # Reset arm height
                    else:
                        print(f"Arm movement blocked: would exceed limits. Current Z: {current_z:.2f}")
                    
            

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        release_all_motors()
        print(f"Pollinated flowers: {robot_pose.pollinated_flowers}")


if __name__ == "__main__":
    # Small CLI options
    show_debug = True
    camera_index = 0
    if len(sys.argv) > 1:
        if sys.argv[1] in ("--debug", "-d"):
            show_debug = True
        else:
            try:
                camera_index = int(sys.argv[1])
            except ValueError:
                pass

    print("Starting combined image-driven multithreaded motor controller.")
    print("Press Ctrl+C to stop.")
    main_loop(camera_index=camera_index, show_debug=show_debug)
