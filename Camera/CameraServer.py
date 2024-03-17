import cv2
import numpy as np
import socket
import struct

# Cấu hình địa chỉ IP và port cho server
SERVER_IP = '192.168.223.132'
PORT = 12345

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_IP, PORT))
server_socket.listen(1)
print("Đang chờ kết nối từ client...")

conn, addr = server_socket.accept()
print(f"Kết nối từ {addr}")

# Khởi tạo camera
cap = cv2.VideoCapture(0)

try:
    while True:
        # Nhận thông điệp từ client
        msg = conn.recv(4) # Giả sử thông điệp 'QUIT' là 4 byte
        if msg.decode('utf-8') == "QUIT":
            print("Client yêu cầu ngắt kết nối.")
            break

        # Đọc hình ảnh từ camera
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc dữ liệu từ camera")
            break

        # Mã hóa hình ảnh sang định dạng .jpg
        result, frame = cv2.imencode('.jpg', frame)
        data = frame.tobytes()

        # Gửi kích thước dữ liệu
        message_size = struct.pack(">Q", len(data))
        conn.sendall(message_size)

        # Gửi dữ liệu hình ảnh
        conn.sendall(data)

except Exception as e:
    print("Lỗi:", e)

finally:
    cap.release()
    conn.close()
    server_socket.close()
