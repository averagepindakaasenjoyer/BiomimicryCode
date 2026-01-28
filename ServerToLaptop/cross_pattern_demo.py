"""
Cross Pattern Movement Demo

Demonstrates a simple cross-shaped movement pattern:
1. Move to center (9, 22.5) - middle of workspace
2. Move forward (Y+ to 18)
3. Return to center
4. Move backward (Y- to 0)
5. Return to center
6. Move left (X+ to 45)
7. Return to center
8. Move right (X- to 0)
9. Return to origin

Workspace limits:
- X (rails): 0-45 cm (0=right, 45=left)
- Y (main): 0-18 cm (0=rear, 18=front)
- Z (arm): 0-20 cm (0=down, 20=up)
"""

import socket
import pickle
import time
import sys
import struct
import threading

# Network Configuration
PI_IP = "100.98.87.47"
PORT = 8000

# Motor Parameters
WHEEL_DIAMETER_CM = 2.5
CIRCUMFERENCE_CM = WHEEL_DIAMETER_CM * 3.14159
STEPS_PER_REV = 200
STEPS_PER_CM = STEPS_PER_REV / CIRCUMFERENCE_CM
scale_move = 0.1

# Workspace limits
LIMIT_X_MIN = 0.0       # Rightmost
LIMIT_X_MAX = 45.0      # Leftmost
LIMIT_Y_MIN = 0.0       # Rearmost
LIMIT_Y_MAX = 18.0      # Frontmost

# Current position tracking
current_pos = {"x": 0.0, "y": 0.0}

# Frame reception control
stop_frame_thread = False

def frame_reception_thread(sock):
    """Continuously receive and discard frames from Pi to prevent blocking."""
    global stop_frame_thread
    data = b""
    payload_size = struct.calcsize("Q")
    frame_count = 0
    
    print("[CROSS DEMO] Frame reception thread started")
    
    try:
        while not stop_frame_thread:
            # Receive frame size header
            while len(data) < payload_size and not stop_frame_thread:
                try:
                    packet = sock.recv(4096)
                    if not packet:
                        print("[CROSS DEMO] Frame reception: connection closed by Pi")
                        return
                    data += packet
                except socket.timeout:
                    continue
                except Exception as e:
                    if not stop_frame_thread:
                        print(f"[CROSS DEMO] Frame reception header error: {e}")
                    return
            
            if stop_frame_thread:
                break
                
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]
            
            # Receive frame data
            while len(data) < msg_size and not stop_frame_thread:
                try:
                    packet = sock.recv(4096)
                    if not packet:
                        print("[CROSS DEMO] Frame reception: connection closed by Pi")
                        return
                    data += packet
                except socket.timeout:
                    continue
                except Exception as e:
                    if not stop_frame_thread:
                        print(f"[CROSS DEMO] Frame reception data error: {e}")
                    return
            
            if stop_frame_thread:
                break
            
            # Discard frame data (we don't need it for this demo)
            data = data[msg_size:]
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"[CROSS DEMO] Frame reception: received {frame_count} frames")
            
    except Exception as e:
        if not stop_frame_thread:
            print(f"[CROSS DEMO] Frame reception error: {e}")
    finally:
        print("[CROSS DEMO] Frame reception thread exiting")

def connect_to_pi():
    """Connect to Pi server."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((PI_IP, PORT))
        print(f"[CROSS DEMO] Connected to Pi at {PI_IP}:{PORT}")
        return sock
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return None

def send_command(sock, command_dict):
    """Send motor command to Pi."""
    try:
        data = pickle.dumps(command_dict)
        sock.sendall(data)
        print(f"[CROSS DEMO] Command sent: {command_dict}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send command: {e}")
        raise  # Re-raise to stop demo on communication error

def move_to_position(sock, target_x, target_y, move_speed_cm=0.5):
    """Move to target position with small steps."""
    global current_pos
    
    print(f"\n[CROSS DEMO] Moving from ({current_pos['x']:.1f}, {current_pos['y']:.1f}) to ({target_x:.1f}, {target_y:.1f})")
    
    while True:
        # Calculate remaining distance
        dx = target_x - current_pos["x"]
        dy = target_y - current_pos["y"]
        
        dist = (dx**2 + dy**2)**0.5
        
        if dist < 0.1:  # Close enough to target
            print(f"[CROSS DEMO] Reached target ({target_x:.1f}, {target_y:.1f})")
            break
        
        # Normalize and apply speed
        if dist > 0:
            move_x = (dx / dist) * move_speed_cm
            move_y = (dy / dist) * move_speed_cm
        else:
            move_x = move_y = 0
        
        # Convert to motor steps
        move_plan = {}
        actual_move_x = 0.0
        actual_move_y = 0.0
        
        if abs(move_x) >= 0.05:
            if move_x > 0:  # Move left
                steps = int(move_x * STEPS_PER_CM * scale_move)
                if steps > 0:
                    move_plan["rails"] = steps
                    actual_move_x = steps / (STEPS_PER_CM * scale_move)
            else:  # Move right
                steps = int(abs(move_x) * STEPS_PER_CM * scale_move)
                if steps > 0:
                    move_plan["rails"] = -steps
                    actual_move_x = -steps / (STEPS_PER_CM * scale_move)
        
        if abs(move_y) >= 0.05:
            if move_y > 0:  # Move forward
                steps = int(move_y * STEPS_PER_CM * scale_move)
                if steps > 0:
                    move_plan["main"] = steps
                    actual_move_y = steps / (STEPS_PER_CM * scale_move)
            else:  # Move backward
                steps = int(abs(move_y) * STEPS_PER_CM * scale_move)
                if steps > 0:
                    move_plan["main"] = -steps
                    actual_move_y = -steps / (STEPS_PER_CM * scale_move)
        
        if move_plan:
            send_command(sock, move_plan)
            # Update position based on actual steps sent, not desired movement
            current_pos["x"] += actual_move_x
            current_pos["y"] += actual_move_y
            
            # Clamp to limits
            current_pos["x"] = max(LIMIT_X_MIN, min(LIMIT_X_MAX, current_pos["x"]))
            current_pos["y"] = max(LIMIT_Y_MIN, min(LIMIT_Y_MAX, current_pos["y"]))
            
            print(f"  -> Current position: ({current_pos['x']:.1f}, {current_pos['y']:.1f})")
        
        time.sleep(0.2)

def reset_motors(sock):
    """Reset all motors."""
    print("[CROSS DEMO] Resetting motors...")
    send_command(sock, {"rails": 0, "main": 0, "arm": 0})
    time.sleep(1.0)

def cross_pattern_demo(sock):
    """Execute cross pattern movement."""
    
    print("\n" + "="*60)
    print("CROSS PATTERN MOVEMENT DEMO")
    print("="*60)
    print("\nWorkspace:")
    print("  X (rails): 0-45 cm (0=right, 45=left)")
    print("  Y (main):  0-18 cm (0=rear, 18=front)")
    print("  Origin:    (0, 0) = rear-right corner")
    print("\nMovement sequence:")
    print("  1. Move to center (22.5, 9)")
    print("  2. Move forward (Y+)")
    print("  3. Return to center")
    print("  4. Move backward (Y-)")
    print("  5. Return to center")
    print("  6. Move left (X+)")
    print("  7. Return to center")
    print("  8. Move right (X-)")
    print("  9. Return to origin")
    print("="*60)
    
    try:
        # Reset first
        reset_motors(sock)
        current_pos["x"] = 0.0
        current_pos["y"] = 0.0
        
        time.sleep(1.0)
        
        center_x = 22.5
        center_y = 9.0
        
        # 1. Move to center
        print("\n[STEP 1] Moving to center...")
        move_to_position(sock, center_x, center_y)
        time.sleep(1.0)
        
        # 2. Move forward (Y increases to 18)
        print("\n[STEP 2] Moving forward (Y+ to 18)...")
        move_to_position(sock, center_x, 18.0)
        time.sleep(1.0)
        
        # 3. Return to center
        print("\n[STEP 3] Returning to center...")
        move_to_position(sock, center_x, center_y)
        time.sleep(1.0)
        
        # 4. Move backward (Y decreases to 0)
        print("\n[STEP 4] Moving backward (Y- to 0)...")
        move_to_position(sock, center_x, 0.0)
        time.sleep(1.0)
        
        # 5. Return to center
        print("\n[STEP 5] Returning to center...")
        move_to_position(sock, center_x, center_y)
        time.sleep(1.0)
        
        # 6. Move left (X increases to 45)
        print("\n[STEP 6] Moving left (X+ to 45)...")
        move_to_position(sock, 45.0, center_y)
        time.sleep(1.0)
        
        # 7. Return to center
        print("\n[STEP 7] Returning to center...")
        move_to_position(sock, center_x, center_y)
        time.sleep(1.0)
        
        # 8. Move right (X decreases to 0)
        print("\n[STEP 8] Moving right (X- to 0)...")
        move_to_position(sock, 0.0, center_y)
        time.sleep(1.0)
        
        # 9. Return to origin
        print("\n[STEP 9] Returning to origin (0, 0)...")
        move_to_position(sock, 0.0, 0.0)
        time.sleep(1.0)
        
        print("\n" + "="*60)
        print("CROSS PATTERN DEMO COMPLETE!")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n[CROSS DEMO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
    finally:
        # Release motors at end
        print("[CROSS DEMO] Releasing motors...")
        send_command(sock, {"_action": "release_all"})

def main():
    """Main function."""
    global stop_frame_thread
    
    print("[CROSS DEMO] Cross Pattern Movement Demonstration\n")
    
    sock = connect_to_pi()
    if sock is None:
        print("[ERROR] Could not connect to Pi. Exiting.")
        sys.exit(1)
    
    # Set socket timeout for robustness
    sock.settimeout(5.0)
    
    # Start frame reception thread to prevent Pi from blocking
    print("[CROSS DEMO] Starting frame reception thread...")
    frame_thread = threading.Thread(target=frame_reception_thread, args=(sock,), daemon=True)
    frame_thread.start()
    
    # Give frame thread time to start and receive first frames
    print("[CROSS DEMO] Waiting for frame reception to stabilize...")
    time.sleep(2.0)
    print("[CROSS DEMO] Ready to begin movement\n")
    
    try:
        cross_pattern_demo(sock)
    except KeyboardInterrupt:
        print("\n[CROSS DEMO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[CROSS DEMO] Shutting down...")
        stop_frame_thread = True
        time.sleep(0.5)
        try:
            sock.close()
        except:
            pass
        print("[CROSS DEMO] Disconnected from Pi")

if __name__ == "__main__":
    main()
