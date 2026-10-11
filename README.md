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
│       └── visualizer.py          # Giao diện HUD trực quan & Bảng đối chiếu trực tiếp (Unified vs MediaPipe)
├── tools/
│   ├── data_prep/                 # Các công cụ trích xuất frame, gán nhãn tự động và tăng cường dữ liệu
│   ├── train_eye_classifier.py    # Huấn luyện mạng CNN phân loại mắt trên bộ MRL Eye Dataset
│   └── train_unified_yolo.py      # Làm sạch nhãn mí mắt và huấn luyện gộp YOLOv8-Pose (YawDD + DDD)
├── config.json                    # Cấu hình trung tâm của toàn hệ thống
├── main.py                        # Điểm khởi chạy duy nhất của chương trình
└── requirements.txt               # Danh sách thư viện phụ thuộc
```

---

## 2. Kiến trúc AI & Xử lý Ảnh số Nổi bật

1. **Kiến trúc Nhận diện Lai Thông minh (`mediapipe` - Mặc định):**
   - **Định vị điểm mốc 3D (`MediaPipe FaceLandmarker`):** Bám sát 478 điểm mốc 3D và trích xuất hệ số cơ mặt 3D (`eyeBlinkLeft`, `eyeBlinkRight`).
   - **Mạng CNN Phân loại mắt Tầng 2 (`models/eye_classifier.pt`):** Cắt vùng mắt từ tọa độ điểm mốc, cân bằng tương phản cục bộ **CLAHE + Z-score** và đưa qua mạng `EyeStateNet` (huấn luyện trên 84.898 ảnh MRL, đạt **98.5% Accuracy**) để loại bỏ hoàn toàn báo động giả.
   - **Cứu hộ Tự động (`YOLO-Rescue Fallback` với `models/dms_unified.pt`):** Khi tài xế cúi gục sâu hoặc quay đầu mạnh khiến MediaPipe mất dấu khuôn mặt, mô hình `dms_unified.pt` tự động nhảy vào tiếp quản ngay lập tức.
2. **Tiền xử lý Ảnh số Thích nghi Ngược sáng (`AdaptiveImageProcessor`):**
   - Đo sáng trực tiếp trên **khung khuôn mặt (Face-ROI Metering)** thay vì toàn cảnh để phát hiện tài xế bị tối mặt khi ngược nắng, tự động kích hoạt **Hiệu chỉnh Gamma (LUT)** và **CLAHE trên kênh Luminance (LAB)**.

---

## 3. Hướng dẫn Chạy & Phím tắt

```powershell
python main.py
```

### Phím tắt trực tiếp trên màn hình:
* **`[M]`**: Chuyển đổi tức thì giữa 3 chế độ:
  1. `MEDIAPIPE` (Mặc định: `MediaPipe 3D + EyeCNN + YOLO-Rescue`)
  2. `UNIFIED` (`Model YOLOv8-Pose hợp nhất dms_unified.pt + EyeCNN`)
  3. `COMPARE` (**Chế độ Đối chiếu Trực tiếp** 2 mô hình trên cùng một màn hình)
* **`[V]`**: Đổi nhanh giữa Camera Laptop (`0`) và DroidCam Điện thoại (`1`)
* **`[R]`**: Xoay khung hình Camera (`0°` $\rightarrow$ `90°` $\rightarrow$ `180°` $\rightarrow$ `270°`)
* **`[C]`**: Hiệu chuẩn lại (Auto-Calibration 5 giây) theo khuôn mặt người lái
* **`[P]`**: Đổi chế độ Tiền xử lý Ảnh số (`AUTO` $\rightarrow$ `CLAHE` $\rightarrow$ `OFF`)
* **`[Q]` / `ESC`**: Thoát chương trình
