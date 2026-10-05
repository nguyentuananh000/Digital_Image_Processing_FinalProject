from ultralytics import YOLO
import cv2

# Nạp file trọng số bạn vừa copy sang (đảm bảo file best.pt nằm cùng thư mục với code này)
model = YOLO('best.pt')

# Khởi động Webcam (Số 0 là camera mặc định)
cap = cv2.VideoCapture(0)

print("Đang khởi động Camera... Bấm phím 'q' để thoát.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Không thể đọc từ webcam.")
        break

    # Phân tích khung hình
    results = model(frame, verbose=False)
    
    # Vẽ khung và 16 điểm mốc lên ảnh
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("DMS - YOLOv8 Pose Live Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()