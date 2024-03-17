import cv2
import numpy as np
import socket
import struct
import threading

class Server:
    def __init__(self, ip_address='192.168.223.132', port=12345):
        self.ip_address = ip_address
        self.port = port
    
    def start(self):
        print("Server is starting...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.ip_address, self.port))
            server_socket.listen(1)
            print("Waiting for connection from client...")
            conn, addr = server_socket.accept()
            with conn:
                print(f"Connected by {addr}")
                handler = ConnectionHandler(conn)
                handler.handle_connection()

class ConnectionHandler:
    def __init__(self, conn):
        self.conn = conn
        self.is_sending = False

    def handle_connection(self):
        command_receiver = CommandReceiver(self.conn, self)
        command_receiver.start()
        
        image_streamer = ImageStreamer(self.conn, self)
        image_streamer.stream_images()

class CommandReceiver(threading.Thread):
    def __init__(self, conn, handler):
        super().__init__()
        self.conn = conn
        self.handler = handler
    
    def run(self):
        try:
            while True:
                data = self.conn.recv(4)
                if not data:
                    break
                command = data.decode('utf-8')
                if command == 'QUIT':
                    print('Client requested disconnection.')
                    self.handler.is_sending = False
                    break
                elif command == 'STAR':
                    print('Starting to send images to client.')
                    self.handler.is_sending = True
        except Exception as e:
            print(f"Error receiving command from client: {e}")

class ImageStreamer:
    def __init__(self, conn, handler):
        self.conn = conn
        self.handler = handler

    def stream_images(self):
        cap = cv2.VideoCapture(0)
        try:
            while True:
                if not self.handler.is_sending:
                    print("Stopped sending images.")
                    break  # Thoát khỏi vòng lặp nếu is_sending == False
                
                ret, frame = cap.read()
                if not ret:
                    print("Could not read data from camera")
                    continue
                _, frame = cv2.imencode('.jpg', frame)
                data = frame.tobytes()
                message_size = struct.pack(">Q", len(data))
                try:
                    self.conn.sendall(message_size)
                    self.conn.sendall(data)
                except Exception as e:
                    print("Connection closed by client: ", e)
                    break
        except Exception as e:
            print(f"Error streaming image: {e}")
        finally:
            cap.release()         

if __name__ == "__main__":
    server = Server()
    server.start()
