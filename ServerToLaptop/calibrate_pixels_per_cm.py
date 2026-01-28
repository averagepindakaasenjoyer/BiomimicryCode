"""
Calibrate PIXELS_PER_CM for laptop_client_manual.py

Instructions:
1. Place an object of known length in front of the camera
2. Run this script and connect to Pi
3. Click on two points on the object (start and end)
4. Enter the real-world distance in cm
5. Script will calculate PIXELS_PER_CM
"""

import socket
import cv2
import pickle
import struct
import numpy as np
import time

# =============================
# Configuration
# =============================
PI_IP = "100.98.87.47"  # Replace with your Pi's IP
PORT = 8000

# Modes
MODE_CALIBRATE = "calibrate"  # Calibrate pixels per cm
MODE_MEASURE = "measure"      # Measure pixel distance from point to center
CURRENT_MODE = MODE_MEASURE  # Set to MODE_MEASURE to measure deltas instead

# Message type identifiers (same as main script)
MSG_TYPE_FRAME = 1
MSG_TYPE_COMMAND = 2

# Camera orientation (same as main script)
SWAP_CAMERAS = True
ROTATE_LEFT = 0 if SWAP_CAMERAS else 180
ROTATE_RIGHT = 180 if SWAP_CAMERAS else 0

# =============================
# Global Variables
# =============================
points = []
current_frame = None

def rotate_frame(frame, rotation_degrees):
    """Rotate frame by specified degrees."""
    if rotation_degrees == 0:
        return frame
    elif rotation_degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return frame

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to select points."""
    global points, current_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if CURRENT_MODE == MODE_CALIBRATE:
            # Calibration mode: need 2 points
            if len(points) < 2:
                points.append((x, y))
                print(f"Point {len(points)} selected: ({x}, {y})")
                
                # Draw point on frame
                if current_frame is not None:
                    display = current_frame.copy()
                    for i, pt in enumerate(points):
                        cv2.circle(display, pt, 5, (0, 255, 0), -1)
                        cv2.putText(display, f"P{i+1}", (pt[0]+10, pt[1]-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if len(points) == 2:
                        # Draw line between points
                        cv2.line(display, points[0], points[1], (0, 255, 255), 2)
                        
                        # Calculate pixel distance
                        dx = points[1][0] - points[0][0]
                        dy = points[1][1] - points[0][1]
                        pixel_distance = np.sqrt(dx**2 + dy**2)
                        
                        # Display info
                        mid_x = (points[0][0] + points[1][0]) // 2
                        mid_y = (points[0][1] + points[1][1]) // 2
                        cv2.putText(display, f"{pixel_distance:.1f} px", (mid_x, mid_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    cv2.imshow("Calibration - Left Camera", display)
        
        elif CURRENT_MODE == MODE_MEASURE:
            # Measure mode: click once to get delta to center
            h, w = current_frame.shape[:2]
            center_x = w // 2
            center_y = h // 2
            
            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx**2 + dy**2)
            
            print(f"\nPoint clicked: ({x}, {y})")
            print(f"Frame center: ({center_x}, {center_y})")
            print(f"Delta from center:")
            print(f"  X (horizontal):  {dx:+.1f} pixels")
            print(f"  Y (vertical):    {dy:+.1f} pixels")
            print(f"  Total distance:  {distance:.1f} pixels")
            print()

def receive_stereo_frames(client_socket, data, payload_size):
    """Receive stereo frame pair from Pi."""
    # Read header: 8 bytes (size) + 1 byte (type)
    header_size = payload_size + 1
    while len(data) < header_size:
        packet = client_socket.recv(4096)
        if not packet:
            return None, None, data
        data += packet
    
    packed_msg_size = data[:payload_size]
    msg_type = data[payload_size]
    data = data[header_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]
    
    # Verify this is a frame message
    if msg_type != MSG_TYPE_FRAME:
        print(f"[WARNING] Expected frame (type {MSG_TYPE_FRAME}), got type {msg_type}")
        # Skip this message
        while len(data) < msg_size:
            packet = client_socket.recv(4096)
            if not packet:
                return None, None, data
            data += packet
        data = data[msg_size:]
        return None, None, data
    
    while len(data) < msg_size:
        packet = client_socket.recv(4096)
        if not packet:
            return None, None, data
        data += packet
    
    frame_data = data[:msg_size]
    data = data[msg_size:]
    
    frames_dict = pickle.loads(frame_data)
    
    frame_left = cv2.imdecode(frames_dict['left'], cv2.IMREAD_COLOR)
    frame_right = cv2.imdecode(frames_dict['right'], cv2.IMREAD_COLOR)
    
    return frame_left, frame_right, data

def main():
    global current_frame, points
    
    print("\n" + "="*70)
    if CURRENT_MODE == MODE_CALIBRATE:
        print("PIXELS_PER_CM CALIBRATION TOOL")
        print("="*70)
        print("\nInstructions:")
        print("1. Place an object of known length in front of the camera")
        print("2. Click on two points to measure (e.g., start and end of ruler)")
        print("3. Enter the real-world distance between the points in cm")
        print("4. Script will calculate PIXELS_PER_CM for you")
        print("\nPress 'r' to reset points, 'q' to quit without calibrating")
    elif CURRENT_MODE == MODE_MEASURE:
        print("PIXEL DISTANCE MEASUREMENT TOOL")
        print("="*70)
        print("\nInstructions:")
        print("1. Blue crosshair shows frame center")
        print("2. Click on any point to measure its distance from center")
        print("3. X and Y pixel distances will be displayed")
        print("\nPress 'q' to quit")
    print("="*70 + "\n")
    
    # Connect to Pi
    print(f"Connecting to Pi at {PI_IP}:{PORT}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((PI_IP, PORT))
        print(f"Connected!\n")
    except Exception as e:
        print(f"Connection failed: {e}")
        return
    
    # Wait for first frame
    print("Waiting for video frames...")
    data = b""
    payload_size = struct.calcsize("Q")
    
    frame_left = None
    while frame_left is None:
        frame_left, frame_right, data = receive_stereo_frames(client_socket, data, payload_size)
        if frame_left is None:
            time.sleep(0.1)
    
    print("Frames received! Opening window...\n")
    
    # Setup window and mouse callback
    cv2.namedWindow("Calibration - Left Camera")
    cv2.setMouseCallback("Calibration - Left Camera", mouse_callback)
    
    try:
        while True:
            # Receive new frame
            frame_left, frame_right, data = receive_stereo_frames(client_socket, data, payload_size)
            
            if frame_left is None:
                print("Failed to receive frame")
                break
            
            # Apply rotation
            frame_left = rotate_frame(frame_left, ROTATE_LEFT)
            current_frame = frame_left.copy()
            h, w = current_frame.shape[:2]
            center_x = w // 2
            center_y = h // 2
            
            # Display frame with points/crosshair
            display = current_frame.copy()
            
            if CURRENT_MODE == MODE_CALIBRATE:
                # Calibration mode: show measurement between 2 points
                for i, pt in enumerate(points):
                    cv2.circle(display, pt, 5, (0, 255, 0), -1)
                    cv2.putText(display, f"P{i+1}", (pt[0]+10, pt[1]-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if len(points) == 2:
                    # Draw line between points
                    cv2.line(display, points[0], points[1], (0, 255, 255), 2)
                    
                    # Calculate pixel distance
                    dx = points[1][0] - points[0][0]
                    dy = points[1][1] - points[0][1]
                    pixel_distance = np.sqrt(dx**2 + dy**2)
                    
                    # Display info
                    mid_x = (points[0][0] + points[1][0]) // 2
                    mid_y = (points[0][1] + points[1][1]) // 2
                    cv2.putText(display, f"{pixel_distance:.1f} pixels", (mid_x, mid_y-20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display, "Press ENTER to input real distance", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display, f"Click {2-len(points)} more point(s)", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            elif CURRENT_MODE == MODE_MEASURE:
                # Measure mode: show center crosshair
                # Draw crosshair at center
                crosshair_size = 30
                cv2.line(display, (center_x - crosshair_size, center_y), 
                        (center_x + crosshair_size, center_y), (255, 0, 0), 2)
                cv2.line(display, (center_x, center_y - crosshair_size), 
                        (center_x, center_y + crosshair_size), (255, 0, 0), 2)
                cv2.circle(display, (center_x, center_y), 8, (255, 0, 0), 2)
                
                # Display center coordinates
                cv2.putText(display, f"Center: ({center_x}, {center_y})", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, "Click any point to measure delta", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Calibration - Left Camera", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                if CURRENT_MODE == MODE_CALIBRATE:
                    print("\nCalibration cancelled")
                else:
                    print("\nMeasurement cancelled")
                break
            
            elif key == ord('r') and CURRENT_MODE == MODE_CALIBRATE:
                points = []
                print("\nPoints reset")
            
            elif key == 13 and len(points) == 2 and CURRENT_MODE == MODE_CALIBRATE:  # Enter key
                # Calculate pixel distance
                dx = points[1][0] - points[0][0]
                dy = points[1][1] - points[0][1]
                pixel_distance = np.sqrt(dx**2 + dy**2)
                
                print("\n" + "="*70)
                print(f"Point 1: {points[0]}")
                print(f"Point 2: {points[1]}")
                print(f"Pixel distance: {pixel_distance:.2f} pixels")
                print(f"  Horizontal (X): {abs(dx):.2f} pixels")
                print(f"  Vertical (Y): {abs(dy):.2f} pixels")
                print("="*70)
                
                # Get real-world distance
                try:
                    real_distance_cm = float(input("\nEnter real-world distance in cm: "))
                    
                    if real_distance_cm <= 0:
                        print("[ERROR] Distance must be positive!")
                        continue
                    
                    # Calculate PIXELS_PER_CM
                    pixels_per_cm = pixel_distance / real_distance_cm
                    
                    print("\n" + "="*70)
                    print("CALIBRATION RESULTS")
                    print("="*70)
                    print(f"  Real distance:     {real_distance_cm:.2f} cm")
                    print(f"  Pixel distance:    {pixel_distance:.2f} pixels")
                    print(f"\n  PIXELS_PER_CM =    {pixels_per_cm:.2f}")
                    print("\n  Update this value in laptop_client_manual.py:")
                    print(f"    PIXELS_PER_CM = {pixels_per_cm:.2f}")
                    print("="*70 + "\n")
                    
                    # Ask to continue
                    cont = input("Calibrate another measurement? (y/n): ").lower()
                    if cont != 'y':
                        break
                    else:
                        points = []
                        print("\nPoints reset. Click two new points.")
                
                except ValueError:
                    print("[ERROR] Invalid input! Please enter a number.")
                except KeyboardInterrupt:
                    print("\nCalibration cancelled")
                    break
    
    except KeyboardInterrupt:
        print("\nCancelled")
    
    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print("Cleanup complete")

if __name__ == "__main__":
    main()
