import cv2
import numpy as np
import socket
import struct
import threading

# Configure the IP address and port for the server
SERVER_IP = '192.168.223.132'
PORT = 12345

# Initialize the server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_IP, PORT))
server_socket.listen(1)
print("Waiting for connection from client...")

conn, addr = server_socket.accept()
print(f"Connected from {addr}")

# Flag to control the sending of images
is_sending = False

def handle_client(conn):
    global is_sending
    try:
        while True:
            # Receive message from client
            msg = conn.recv(4)
            if not msg:
                break  # End the loop if no data is received
            message = msg.decode('utf-8')
            if message == "QUIT":
                print("Client requested disconnection.")
                is_sending = False
                break
            elif message == "STAR":
                print("Starting to send images to client.")
                is_sending = True
    except Exception as e:
        print("Error receiving command from client:", e)
    finally:
        conn.close()

def send_images(conn):
    global is_sending
    cap = cv2.VideoCapture(0)
    try:
        while True:
            if not is_sending:
                continue  # Wait until there is a command to send images

            # Read image from camera
            ret, frame = cap.read()
            if not ret:
                print("Could not read data from camera")
                continue

            # Encode the image in .png or .jpg format
            result, frame = cv2.imencode('.jpg', frame)
            data = frame.tobytes()

            # Check before sending to avoid 'Bad file descriptor'
            if not is_sending:
                print("Stopped sending images.")
                break

            # Send data size
            message_size = struct.pack(">Q", len(data))
            try:
                conn.sendall(message_size)
                # Send image data
                conn.sendall(data)
            except socket.error as e:
                print("Error sending image:", e)
                break
    except Exception as e:
        print("Undefined error sending image:", e)
    finally:
        cap.release()

# Create and start the client handling thread
client_thread = threading.Thread(target=handle_client, args=(conn,))
client_thread.start()

# Send images in a separate thread
image_thread = threading.Thread(target=send_images, args=(conn,))
image_thread.start()

client_thread.join()
image_thread.join()
server_socket.close()
