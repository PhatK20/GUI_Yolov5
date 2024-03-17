import cv2
import torch
import numpy as np
import time

# Correct the path for your trained weights and the YOLOv5 repository directory
# model_path = 'yolov5/best_model.pt'
model_path = 'yolov5-NTBT-NNT/1best.pt'
# Load the model from the local clone of the YOLOv5 repository
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the default camera
cap = cv2.VideoCapture(0)  # '0' is the default ID for the primary camera

prev_frame_time = 0  # Variable for storing time of previous frame
new_frame_time = 0  # Variable for storing time of new frame

while True:
    ret, frame = cap.read()  # Read each frame from the camera
    if not ret:
        print("Failed to grab frame")
        break

    # Process the frame with YOLOv5
    results = model(frame)

    # Chỉ số của class "bottle" trong danh sách class mà mô hình được huấn luyện để nhận diện
    # bottle_index = model.names.index('bottle')
    # Lấy tọa độ và nhãn của mỗi đối tượng phát hiện được
    for *xyxy, conf, cls in results.xyxy[0]:
        # if int(cls) == bottle_index:
            # Tọa độ của bounding box
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Tính tọa độ tâm của bounding box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
        
            print(f"Tâm của đối tượng: ({x_center}, {y_center})")

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps_text = f'FPS: {fps}'

    # Display FPS on the frame
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the processed results
    cv2.imshow('YOLOv5 Detection', np.squeeze(results.render()))

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
