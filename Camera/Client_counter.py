import socket
import time

def send_messages(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        i = 0
        try:
            while True:
                message = f"Hello World {i}"
                s.sendall(message.encode())
                time.sleep(1)
                i += 1
        except KeyboardInterrupt:
            print("Client stopped.")

if __name__ == "__main__":
    # HOST = '127.0.0.1'  # Server address
    HOST = '192.168.223.1'
    PORT = 65432        # Server port
    send_messages(HOST, PORT)
