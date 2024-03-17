import cv2
import socket
import struct
import pickle

# # Khởi tạo và mở camera
# cap = cv2.VideoCapture(0)

# # Thiết lập kết nối socket đến server
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# host_ip = '192.168.223.1'  # Thay địa chỉ IP của server ở đây
# port = 9999  # Cổng kết nối tới server
# client_socket.connect((host_ip, port))

# while True:
#     ret, frame = cap.read()  # Đọc hình ảnh từ camera
#     if not ret:
#         print("Không thể nhận hình ảnh từ camera. Kiểm tra lại camera.")
#         break
#     # Mã hóa hình ảnh để gửi qua socket
#     data = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
#     # Gửi kích thước dữ liệu trước
#     message_size = struct.pack("L", len(data))
#     client_socket.sendall(message_size + data)

# # Đóng kết nối và camera khi xong
# cap.release()
# client_socket.close()

cap = cv2.VideoCapture(0)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.223.1'  # Địa chỉ IP của server
port = 9999  # Cổng kết nối
client_socket.connect((host_ip, port))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Mã hóa hình ảnh thành PNG
    result, buffer = cv2.imencode('.png', frame)
    if result:
        # Lấy dữ liệu dưới dạng byte và tính kích thước
        data = buffer.tobytes()
        size = len(data)  # Tính số byte của frame

        print(f"Kích thước của frame gửi đi: {size} bytes")
        # Gửi kích thước dữ liệu hình ảnh trước
        client_socket.sendall(struct.pack(">Q", len(data)))
        client_socket.sendall(data)

cap.release()
client_socket.close()