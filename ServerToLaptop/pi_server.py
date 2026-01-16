import socket
import cv2
import pickle
import struct

# Replace with the port you want to use
PORT = 8000

# Set up socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', PORT))
server_socket.listen(1)
print(f"[Pi] Waiting for connection on port {PORT}...")
conn, addr = server_socket.accept()
print(f"[Pi] Connected to {addr}")

# Open camera
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            continue

        # Compress frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        data = pickle.dumps(buffer)
        message = struct.pack("Q", len(data)) + data
        conn.sendall(message)

        # Receive motor command from laptop
        command = conn.recv(1024).decode()
        if command:
            print(f"[Pi] Motor command received: {command}")
            # Here, you would send this command to GPIO/motor driver
except Exception as e:
    print(f"[Pi] Error: {e}")
finally:
    cap.release()
    conn.close()
    server_socket.close()
