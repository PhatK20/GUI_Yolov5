import cv2

def main():
    # Mở camera
    cap = cv2.VideoCapture(0)

    # Kiểm tra xem camera có mở thành công không
    if not cap.isOpened():
        print("Không thể mở camera. Đảm bảo rằng camera đã được kết nối và được phân định làm camera mặc định.")
        return

    while True:
        # Đọc frame từ camera
        ret, frame = cap.read()

        # Kiểm tra xem frame có được đọc thành công không
        if not ret:
            print("Không thể đọc frame từ camera.")
            break

        # Hiển thị frame lên màn hình
        cv2.imshow('Camera', frame)

        # Đợi phím nhấn 'q' để thoát vòng lặp
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera và đóng cửa sổ hiển thị
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
