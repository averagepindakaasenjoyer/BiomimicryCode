"""
Simulation-capable version of your robot movement code.
Drop this file in place of the original to run without Adafruit hardware.
Requires: numpy, opencv-python
"""

import time
import numpy as np
import cv2
import threading

# ----- CONFIG -----
SIMULATE = True        # Set False when switching to real hardware and Adafruit libs
VIDEO_INDEX = 0        # webcam index; if no webcam available synthetic image is used
diameter_wheel = 2.5   # cm
circumference_wheel = diameter_wheel * np.pi
steps_per_revolution = 200
steps_per_cm = steps_per_revolution / circumference_wheel

# conversion used for PIXEL -> CM in get_moving_direction (calibrate for your setup)
PIXELS_PER_CM = 40.0

# small deadzone in pixels
DEADZONE_PX = 20

# step delay
DEFAULT_STEP_SPEED = 0.01

# ----- MOCK / REAL MOTOR SETUP -----
# In simulate mode we provide MockStepper objects that print actions.
# When you want to run on hardware, replace the mock setup with your MotorKit setup
# and keep the rest of the script the same (see notes at the end).

FORWARD = 1
BACKWARD = -1

class MockStepper:
    def __init__(self, name):
        self.name = name
        self.position = 0
        self._lock = threading.Lock()
    def onestep(self, direction=FORWARD, style=None):
        # accept numeric direction or stepper.FORWARD/BACKWARD
        d = 1 if direction == FORWARD else -1
        with self._lock:
            self.position += d
        print(f"[MOCK] {self.name} onestep({d}) -> pos={self.position}")
    def release(self):
        print(f"[MOCK] {self.name} released")

# instantiate motor_dict (same keys you use elsewhere)
motor_dict = {
    "rear_main": MockStepper("rear_main"),
    "front_main": MockStepper("front_main"),
    "right_rail": MockStepper("right_rail"),
    "left_rail": MockStepper("left_rail"),
    "arm": MockStepper("arm"),
}

# mapping of logical directions to physical motors (multiplier indicates sign)
direction_dict = {
    "front": [("rear_main", -1), ("front_main", 1)],
    "rear": [("rear_main", 1), ("front_main", -1)],
    "left": [("left_rail", 1), ("right_rail", -1)],
    "right": [("right_rail", 1), ("left_rail", -1)],
    "up": [("arm", 1)],
    "down": [("arm", -1)],
}

# ----- MOVEMENT PRIMITIVES -----

def move_cm(distance_cm, speed=DEFAULT_STEP_SPEED, motor=None):
    """
    Move a single motor by distance_cm (signed). `motor` is a stepper object.
    If motor is None, default to rear_main mock motor.
    """
    if motor is None:
        motor = motor_dict["rear_main"]
    steps = int(round(abs(distance_cm) * steps_per_cm))
    if steps == 0:
        return
    direction = FORWARD if distance_cm > 0 else BACKWARD
    for _ in range(steps):
        motor.onestep(direction=direction, style=None)
        time.sleep(speed)
    motor.release()

def move_direction(speed=DEFAULT_STEP_SPEED, direction_to_move=None):
    """
    direction_to_move: list of tuples [(direction_name, distance_cm), ...]
    distance_cm may be positive or negative; sign controls net direction.
    This function translates logical directions to physical motors and steps each motor
    the required number of steps (per-motor remaining-step tracking).
    """
    if direction_to_move is None:
        direction_to_move = [("front", 0)]

    # Build per-physical motor required steps and direction
    motor_tasks = {}  # motor_name -> (steps, sign)
    for dir_name, dist_cm in direction_to_move:
        if dir_name not in direction_dict:
            continue
        for motor_name, motor_mult in direction_dict[dir_name]:
            # Effective sign: motor_mult * sign(dist_cm)
            sign = 1 if dist_cm >= 0 else -1
            effective_sign = motor_mult * sign
            steps = int(round(abs(dist_cm) * steps_per_cm))
            if steps == 0:
                continue
            # If motor already in tasks, keep the max steps (simple merging)
            prev = motor_tasks.get(motor_name)
            if prev is None or steps > prev[0]:
                motor_tasks[motor_name] = (steps, effective_sign)

    # Per-motor stepping loop
    remaining = {m: cnt for (m, (cnt, _)) in motor_tasks.items()}
    signs = {m: s for (m, (_, s)) in motor_tasks.items()}

    while any(v > 0 for v in remaining.values()):
        for m in list(remaining.keys()):
            if remaining[m] <= 0:
                continue
            motor_obj = motor_dict.get(m)
            if motor_obj is None:
                continue
            direction = FORWARD if signs[m] > 0 else BACKWARD
            motor_obj.onestep(direction=direction, style=None)
            remaining[m] -= 1
        time.sleep(speed)

    for m in motor_tasks.keys():
        motor_dict[m].release()

# ----- IMAGE PROCESSING -----

def detect_circle(image):
    """
    Detect yellow circle. Returns list of normalized centers [(nx, ny), ...] (0..1).
    Returns empty list if none found.
    """
    recognized = []
    if image is None:
        return recognized

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([18, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    masked = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 7)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    h, w = image.shape[:2]
    min_r = max(10, int(min(h, w) * 0.05))
    max_r = int(min(h, w) * 0.4)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                               dp=1.5,
                               minDist=max(h, w) // 4,
                               param1=100,
                               param2=40,
                               minRadius=min_r,
                               maxRadius=max_r)
    if circles is None:
        return recognized

    circles = np.uint16(np.around(circles))
    # choose the largest by radius
    best = None
    best_r = -1
    for c in circles[0, :]:
        if c[2] > best_r:
            best = c
            best_r = c[2]
    cx, cy = best[0], best[1]
    nx = cx / float(w)
    ny = cy / float(h)
    recognized.append((nx, ny))

    # draw debug markers
    cv2.circle(image, (int(cx), int(cy)), 3, (0, 255, 0), -1)
    cv2.circle(image, (int(cx), int(cy)), int(best_r), (255, 0, 0), 2)

    return recognized

def process_image():
    """
    Capture a frame from webcam if available; otherwise generate a synthetic image.
    Returns (move_distanceX_px, move_distanceY_px).
    """
    cap = cv2.VideoCapture(VIDEO_INDEX)
    ret, frame = cap.read()
    if not ret or frame is None:
        # fallback synthetic image
        w, h = 640, 480
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # place a yellow circle slightly right and down so we get non-zero offsets
        cv2.circle(frame, (int(w*0.65), int(h*0.6)), 40, (0, 255, 255), -1)
        ret = True
    cap.release()

    if not ret:
        print("Failed to capture or generate image")
        return (0, 0)

    circles = detect_circle(frame)
    image_h, image_w = frame.shape[:2]
    if circles:
        nx, ny = circles[0]
        cx_px = int(nx * image_w)
        cy_px = int(ny * image_h)
        center_x = image_w // 2
        center_y = image_h // 2
        move_distanceX = cx_px - center_x
        move_distanceY = cy_px - center_y
        # debug visual
        cv2.imshow("debug", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
        return (move_distanceX, move_distanceY)

    # show frame for debugging if no circle
    cv2.imshow("debug", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass
    return (0, 0)

# ----- LOGIC TO CONVERT PIXEL OFFSETS TO MOTOR ACTIONS -----

def get_moving_direction(DistanceX, DistanceY):
    """
    Convert pixel distances to a movement dictionary with keys only for required movements.
    Returned format matches your main loop checks: e.g.
      {'front': [('rear_main', -2.0), ('front_main', 2.0)], ...}
    Distances are in CM (float) encoded as the second element in the tuples
    and will be used by the main loop as the distance value.
    """
    moving_dict = {}

    # Convert pixel offset to cm (simple linear mapping)
    # NOTE: calibrate PIXELS_PER_CM for your camera/robot geometry
    if abs(DistanceX) > DEADZONE_PX:
        cm_x = DistanceX / PIXELS_PER_CM  # signed: positive means circle is right of center
        # decide whether that corresponds to 'front' or 'rear' for your robot:
        if cm_x > 0:
            moving_dict["front"] = [("rear_main", -abs(cm_x)), ("front_main", abs(cm_x))]
        else:
            moving_dict["rear"] = [("rear_main", abs(cm_x)), ("front_main", -abs(cm_x))]

    if abs(DistanceY) > DEADZONE_PX:
        cm_y = DistanceY / PIXELS_PER_CM
        # positive Y means circle below center; mapping to rails depends on camera orientation
        if cm_y > 0:
            moving_dict["left"] = [("left_rail", abs(cm_y)), ("right_rail", -abs(cm_y))]
        else:
            moving_dict["right"] = [("right_rail", abs(cm_y)), ("left_rail", -abs(cm_y))]

    return moving_dict

# ----- MAIN LOOP (keeps your structure so you don't need to edit other code) -----

if __name__ == "__main__":
    try:
        while True:
            DistanceX, DistanceY = process_image()
            print(f"DistanceX: {DistanceX}, DistanceY: {DistanceY}")
            moving_dict = get_moving_direction(DistanceX, DistanceY)

            threads = []
            if "front" in moving_dict:
                # value used by your original main: pick first tuple's second element as the distance
                t1 = threading.Thread(target=move_direction, args=(0.01, [("front", moving_dict["front"][0][1])]))
                threads.append(t1)
                t1.start()
            if "rear" in moving_dict:
                t2 = threading.Thread(target=move_direction, args=(0.01, [("rear", moving_dict["rear"][0][1])]))
                threads.append(t2)
                t2.start()
            if "left" in moving_dict:
                t3 = threading.Thread(target=move_direction, args=(0.01, [("left", moving_dict["left"][0][1])]))
                threads.append(t3)
                t3.start()
            if "right" in moving_dict:
                t4 = threading.Thread(target=move_direction, args=(0.01, [("right", moving_dict["right"][0][1])]))
                threads.append(t4)
                t4.start()

            for t in threads:
                t.join()

            # small delay between frames
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Interrupted by user, exiting.")
    finally:
        cv2.destroyAllWindows()
