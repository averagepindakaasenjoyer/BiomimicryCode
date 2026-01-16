import socket
import cv2
import pickle
import struct

# Replace with your Pi's IP
PI_IP = "192.168.1.50"
PORT = 8000

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((PI_IP, PORT))
print(f"[Laptop] Connected to Pi at {PI_IP}:{PORT}")

data = b""
payload_size = struct.calcsize("Q")

try:
    while True:
        # Receive message size
        while len(data) < payload_size:
            packet = client_socket.recv(4096)
            if not packet:
                break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Receive frame data
        while len(data) < msg_size:
            data += client_socket.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Deserialize and decode JPEG
        buffer = pickle.loads(frame_data)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        # ----- Vision Processing Placeholder -----
        # Example: Detect average brightness
        brightness = frame.mean()
        print(f"[Laptop] Frame brightness: {brightness:.2f}")

        # Decide motor command
        if brightness > 100:
            command = "FORWARD 50"
        else:
            command = "STOP"

        # Send motor command back to Pi
        client_socket.send(command.encode())
except Exception as e:
    print(f"[Laptop] Error: {e}")
finally:
    client_socket.close()
