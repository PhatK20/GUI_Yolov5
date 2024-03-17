import cv2
import socket
import struct
from pathlib import Path
import torch

# Khởi tạo mô hình YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# def send_frame_to_cpp(frame, host, port):
#     _, buffer = cv2.imencode('.png', frame)
#     data = buffer.tobytes()
#     client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     client_socket.connect((host, port))
#     client_socket.sendall(struct.pack(">L", len(data)) + data)
#     client_socket.close()

def send_frame_to_cpp(frame, host, port):
    try:
        _, buffer = cv2.imencode('.jpg', frame)  # Sử dụng JPEG để giảm kích thước
        data = buffer.tobytes()
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        client_socket.sendall(struct.pack(">L", len(data)) + data)
        # client_socket.close()  # Bỏ comment dòng này nếu muốn giữ kết nối mở
    except Exception as e:
        print(f"Error sending frame: {e}")
    finally:
        client_socket.close()


def main():
    cap = cv2.VideoCapture(0)
    # trên cùng 1 máy tính dùng 'localhost' or '127.0.0.1'
    host = 'localhost'
    port = 12345

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Phát hiện đối tượng với YOLOv5
        results = model(frame)
        
        # Lấy frame với đối tượng được đánh dấu
        frame_with_detections = results.render()[0]
        
        # Gửi frame đó tới chương trình C++
        send_frame_to_cpp(frame_with_detections, host, port)

if __name__ == "__main__":
    main()
