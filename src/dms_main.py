import cv2
import math
from ultralytics import YOLO

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ LOGIC (Có thể tinh chỉnh)
# ==========================================
EAR_THRESHOLD = 0.25      # Ngưỡng nhắm mắt (Dưới mức này coi là nhắm)
MAR_THRESHOLD = 0.50      # Ngưỡng ngáp (Trên mức này coi là ngáp)
SLEEP_FRAMES = 15         # Số khung hình nhắm mắt liên tục để báo động ngủ gật
YAWN_FRAMES = 10          # Số khung hình mở miệng liên tục để báo động ngáp

# ==========================================
# 2. HÀM TOÁN HỌC TÍNH KHOẢNG CÁCH
# ==========================================
def euclidean_dist(p1, p2):
    """Tính khoảng cách đường thẳng giữa 2 điểm (x, y)"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(eye_points):
    """
    Tính chỉ số EAR (Eye Aspect Ratio) cho 6 điểm của mắt.
    Giả định thứ tự điểm (theo MediaPipe chuẩn):
    0: Khóe mắt ngoài | 1, 2: Mí trên | 3: Khóe mắt trong | 4, 5: Mí dưới
    """
    # Chiều dọc (Mí trên - Mí dưới)
    v1 = euclidean_dist(eye_points[1], eye_points[5])
    v2 = euclidean_dist(eye_points[2], eye_points[4])
    # Chiều ngang (Khóe ngoài - Khóe trong)
    h = euclidean_dist(eye_points[0], eye_points[3])
    
    if h == 0: return 0.0
    return (v1 + v2) / (2.0 * h)

def calculate_mar(mouth_points):
    """
    Tính chỉ số MAR (Mouth Aspect Ratio) cho 4 điểm của miệng.
    Giả định thứ tự: 0: Trái | 1: Trên | 2: Phải | 3: Dưới
    """
    # Chiều dọc (Môi trên - Môi dưới)
    v = euclidean_dist(mouth_points[1], mouth_points[3])
    # Chiều ngang (Khóe miệng trái - phải)
    h = euclidean_dist(mouth_points[0], mouth_points[2])
    
    if h == 0: return 0.0
    return v / h

# ==========================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == '__main__':
    # Nạp mô hình AI
    model = YOLO('best.pt')
    cap = cv2.VideoCapture(0)

    # Biến đếm khung hình để báo động
    sleep_counter = 0
    yawn_counter = 0
    alarm_on = False

    print("Hệ thống DMS đang chạy... Bấm 'q' để thoát.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Ép AI phân tích khung hình hiện tại
        results = model(frame, verbose=False)
        
        # Vẽ sẵn các điểm và khung nhận diện của YOLO lên ảnh
        annotated_frame = results[0].plot()

        # Kiểm tra xem AI có tìm thấy khuôn mặt và điểm mốc nào không
        if results[0].keypoints is not None and len(results[0].keypoints.xy[0]) == 16:
            # Rút trích tọa độ 16 điểm ra thành danh sách các cặp (x, y)
            pts = results[0].keypoints.xy[0].cpu().numpy()
            
            # Phân tách tọa độ dựa trên thuật toán convert.py bạn đã tạo
            left_eye_pts = pts[0:6]    # 6 điểm đầu
            right_eye_pts = pts[6:12]  # 6 điểm tiếp theo
            mouth_pts = pts[12:16]     # 4 điểm cuối

            # Tính toán các chỉ số
            ear_left = calculate_ear(left_eye_pts)
            ear_right = calculate_ear(right_eye_pts)
            avg_ear = (ear_left + ear_right) / 2.0
            mar = calculate_mar(mouth_pts)

            # --- LOGIC CẢNH BÁO NGỦ GẬT ---
            if avg_ear < EAR_THRESHOLD:
                sleep_counter += 1
                if sleep_counter >= SLEEP_FRAMES:
                    cv2.putText(annotated_frame, "CANH BAO: NGU GAT!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            else:
                sleep_counter = 0

            # --- LOGIC CẢNH BÁO NGÁP ---
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= YAWN_FRAMES:
                    cv2.putText(annotated_frame, "CANH BAO: NGAP!", (50, 160), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
            else:
                yawn_counter = 0

            # --- HIỂN THỊ THÔNG SỐ LÊN MÀN HÌNH ---
            cv2.putText(annotated_frame, f"EAR (Mat): {avg_ear:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"MAR (Mieng): {mar:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Xuất hình ảnh ra màn hình
        cv2.imshow("Driver Monitoring System", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()