# Hệ Thống Giám Sát Tài Xế Buồn Ngủ Thời Gian Thực (Real-Time DMS)

Dự án môn học **Xử lý Ảnh số (Digital Image Processing)** — Hệ thống phát hiện ngủ gật, chớp mắt kéo dài (Micro-sleep), ngáp, gục đầu và mất tập trung của tài xế ở tốc độ **30 FPS** thời gian thực.

---

## 1. Cấu trúc Dự án (Gọn gàng & Chuẩn hóa)

```text
project/
├── assets/
│   └── alarm.wav                  # Âm thanh báo động khẩn cấp (Cấp độ 2)
├── models/
│   ├── face_landmarker.task       # MediaPipe 3D FaceLandmarker + 3D Blendshapes (Tầng 1 Chính)
│   ├── eye_classifier.pt          # Mạng CNN phân loại trạng thái mắt huấn luyện trên MRL Eye Dataset (98.5% Acc)
│   └── dms_unified.pt             # Model YOLOv8-Pose hợp nhất (YawDD + DDD, mAP50-95 = 97.2%)
├── src/
│   ├── core/
│   │   ├── camera.py              # Luồng đọc Camera/DroidCam độc lập (30 FPS, xoay ảnh, đổi camera nóng)
│   │   ├── image_processing.py    # Tiền xử lý ảnh số thích nghi vùng mặt (Face-ROI Gamma LUT + CLAHE + One-Euro Filter)
│   │   ├── detector.py            # Kiến trúc Lai Đa Tầng (MediaPipe 3D + EyeCNN + YOLO-Rescue + Compare)
│   │   ├── eye_classifier.py      # Mạng CNN EyeStateNet (48x48 CLAHE + Z-score) phân loại mắt Awake/Sleepy
│   │   ├── math_utils.py          # Công thức EAR (Lai Khoảng cách + Diện tích đa giác), MAR, SolvePnP 3D Head Pose
│   │   ├── tracker_logic.py       # Hiệu chuẩn cá nhân 5s, bù góc cúi đầu Pitch, phân biệt Nói chuyện vs Ngáp, PERCLOS
│   │   └── alert_system.py        # Cảnh báo âm thanh đa cấp (Level 1: Beep 880Hz nhắc nhở, Level 2: Còi báo động)
│   └── ui/
│       ├── app_gui.py             # Giao diện Dashboard hiện đại (CustomTkinter) với đầy đủ thanh điều khiển, đồng hồ đo & nhật ký
│       └── visualizer.py          # Giao diện HUD trực quan & Bảng đối chiếu trực tiếp (Unified vs MediaPipe)
├── tools/
│   ├── data_prep/                 # Các công cụ trích xuất frame, gán nhãn tự động và tăng cường dữ liệu
│   ├── train_eye_classifier.py    # Huấn luyện mạng CNN phân loại mắt trên bộ MRL Eye Dataset
│   └── train_unified_yolo.py      # Làm sạch nhãn mí mắt và huấn luyện gộp YOLOv8-Pose (YawDD + DDD)
├── Bao_Cao_Do_An_DMS.docx         # Bản Báo cáo Đồ án Toàn diện xuất bản file Word (10 chương chuẩn học thuật)
├── config.json                    # Cấu hình trung tâm của toàn hệ thống
├── main.py                        # Điểm khởi chạy duy nhất của chương trình (Tự động mở Dashboard GUI)
└── requirements.txt               # Danh sách thư viện phụ thuộc

---

## 2. Kiến trúc AI & Xử lý Ảnh số Nổi bật

1. **Kiến trúc Nhận diện Lai Thông minh (`mediapipe` - Mặc định):**
   - **Định vị điểm mốc 3D (`MediaPipe FaceLandmarker`):** Bám sát 478 điểm mốc 3D và trích xuất hệ số cơ mặt 3D (`eyeBlinkLeft`, `eyeBlinkRight`).
   - **Mạng CNN Phân loại mắt Tầng 2 (`models/eye_classifier.pt`):** Cắt vùng mắt từ tọa độ điểm mốc, cân bằng tương phản cục bộ **CLAHE + Z-score** và đưa qua mạng `EyeStateNet` (huấn luyện trên 84.898 ảnh MRL, đạt **98.5% Accuracy**) để loại bỏ hoàn toàn báo động giả.
   - **Cứu hộ Tự động (`YOLO-Rescue Fallback` với `models/dms_unified.pt`):** Khi tài xế cúi gục sâu hoặc quay đầu mạnh khiến MediaPipe mất dấu khuôn mặt, mô hình `dms_unified.pt` tự động nhảy vào tiếp quản ngay lập tức.
2. **Tiền xử lý Ảnh số Thích nghi Ngược sáng (`AdaptiveImageProcessor`):**
   - Đo sáng trực tiếp trên **khung khuôn mặt (Face-ROI Metering)** thay vì toàn cảnh để phát hiện tài xế bị tối mặt khi ngược nắng, tự động kích hoạt **Hiệu chỉnh Gamma (LUT)** và **CLAHE trên kênh Luminance (LAB)**.

---

## 3. Giao diện Người dùng Đồ họa (Modern Dashboard) & Hướng dẫn Sử dụng

### 3.1 Khởi chạy Chương trình

* **Chạy giao diện đồ họa trực quan (Khuyên dùng):**
  ```powershell
  python main.py
  ```
* **Chạy chế độ dòng lệnh OpenCV truyền thống:**
  ```powershell
  python main.py --cli
  ```

### 3.2 Các Tính năng trên Dashboard Điều khiển:
* **Khung Video Trực tiếp:** Hiển thị luồng camera 30 FPS mượt mà cùng lớp phủ nhận diện khuôn mặt, các điểm mốc mắt/miệng và HUD trạng thái.
* **Thanh tác vụ nhanh (Quick Action Bar):**
  * `Tạm dừng / Tiếp tục`: Tạm dừng hệ thống bất kỳ lúc nào.
  * `Hiệu chuẩn lại`: Bật chu kỳ 5 giây học ngưỡng sinh trắc riêng cho khuôn mặt tài xế.
  * `Xoay camera`: Xoay khung hình 0°, 90°, 180°, 270° phù hợp khi dùng DroidCam điện thoại.
  * `Tắt âm`: Bật/Tắt chuông báo động khi đang thử nghiệm trong phòng yên tĩnh.
  * `Chụp ảnh`: Lưu ảnh chứng cứ khi phát hiện ngủ gật/ngáp vào thư mục `snapshots/`.
* **Cài đặt Thiết bị & AI (Device & AI Settings):**
  * Danh sách chọn Camera: Webcam Laptop (0), DroidCam Điện thoại (1), hoặc URL IP Cam.
  * Chuyển đổi Model AI tức thì: `MediaPipe 3D (Khuyên dùng)`, `Unified YOLO`, hoặc `Đối chiếu So sánh (Compare)`.
  * Bộ lọc Tiền xử lý Ảnh số: `Tự động thích nghi (Auto)`, `Bật CLAHE`, hoặc `Tắt lọc`.
* **Thanh đo Sinh trắc Thời gian thực (Live Biometrics Meters):**
  * Đồng hồ chỉ số **EAR** (Eye Aspect Ratio), **EyeCNN** (Xác suất mắt nhắm), **MAR** (Mouth Aspect Ratio), và **PERCLOS** (% thời gian nhắm mắt).
* **Thanh trượt Tinh chỉnh Độ nhạy (Real-time Sliders):**
  * Kéo trượt điều chỉnh ngay lập tức: Ngưỡng EAR, Ngưỡng MAR, Thời gian ngủ gật (giây), Thời gian ngáp (giây).
* **Nhật ký Sự kiện Tự động Cuộn (Event Log):**
  * Ghi lại chi tiết thời gian và diễn biến từng lần tài xế buồn ngủ, ngủ gật (Micro-sleep) hay ngáp theo thời gian thực.
