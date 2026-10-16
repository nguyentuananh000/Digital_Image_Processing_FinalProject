import datetime
import json
import os
import sys
import time
from collections import deque
from typing import Optional

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from ..core.alert_system import AlertSystem
from ..core.camera import WebcamStream
from ..core.detector import AsyncDetectorWorker, FaceLandmarkDetector
from ..core.image_processing import AdaptiveImageProcessor
from ..core.tracker_logic import DriverStatus, DrowsinessTracker
from .visualizer import DMSVisualizer

# Cấu hình giao diện CustomTkinter chuẩn Dark Mode
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class DMSApp(ctk.CTk):
    """
    Ứng dụng Giao diện Điều khiển Giám sát Tài xế Thời gian thực (DMS Dashboard).
    - Tự động co giãn thích nghi đa độ phân giải màn hình (Adaptive Multi-Resolution).
    - Render luồng Video hiệu năng cao bằng Direct ImageTk (>60 FPS, tiết kiệm CPU).
    - Hiển thị trực quan mốc khuôn mặt sạch sẽ (Clean Stream), hỗ trợ bật/tắt HUD chữ.
    - Điều khiển trực quan bằng chuột: Bắt đầu/Dừng, Chọn Camera, Chuyển đổi Model AI.
    - Đồng hồ đo và thanh chỉ số sinh trắc trực tiếp (EAR, EyeCNN, MAR, PERCLOS).
    - Thanh trượt tinh chỉnh độ nhạy theo thời gian thực (Real-time Sliders).
    - Nhật ký sự kiện và cảnh báo tự động cuộn (Auto-scroll Event Log).
    """

    def __init__(self, config_path: str = "config.json"):
        super().__init__()

        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config_path = os.path.join(self.root_dir, config_path)
        self.config = self._load_config()

        # Thiết lập cửa sổ chính thích nghi đa độ phân giải màn hình (Responsive Geometry)
        self.title("Hệ Thống Giám Sát Tài Xế (DMS) - Digital Image Processing & AI")
        self._setup_responsive_window()

        # Trạng thái hệ thống
        self.is_running = True
        self.is_muted = False
        self.show_hud_text = False  # Mặc định hiển thị video sạch, không bị hộp chữ che khuôn mặt
        self.fps_history = deque(maxlen=25)
        self.prev_time = time.perf_counter()
        self.last_status_level = 0
        self.last_face_bbox = None
        self._current_photo_img = None  # Giữ tham chiếu ảnh để tránh garbage collector xóa nhầm
        self._last_metrics_update = 0.0
        self._last_ui_alert_level = -1
        self._last_ui_calib_state = False
        self._last_ui_status_text = ""
        self._video_text_cleared = False

        # Các đối tượng mô-đun lõi (sẽ khởi tạo sau khi cửa sổ GUI xuất hiện)
        self.camera = None
        self.img_processor = None
        self.detector = None
        self.ai_worker = None
        self.tracker = None
        self.alert = None

        # Xây dựng giao diện đồ họa tức thì (<0.2s)
        self._build_ui()

        # Hiển thị cửa sổ ngay lập tức lên màn hình cho người dùng
        self.update()

        # Xử lý đóng cửa sổ an toàn
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Phím tắt bàn phím
        self.bind("<space>", lambda e: self._toggle_running())
        self.bind("<c>", lambda e: self._trigger_calibration())
        self.bind("<C>", lambda e: self._trigger_calibration())
        self.bind("<r>", lambda e: self._rotate_camera())
        self.bind("<R>", lambda e: self._rotate_camera())
        self.bind("<h>", lambda e: self._toggle_hud_overlay())
        self.bind("<H>", lambda e: self._toggle_hud_overlay())

        # Khởi động mô-đun ngầm sau 30ms (không chặn cửa sổ hiển thị)
        self.after(30, self._start_system)

    def _setup_responsive_window(self):
        """Tự động tính toán kích thước cửa sổ phù hợp với độ phân giải màn hình thực tế của người dùng."""
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        # Trừ khoảng đệm taskbar và viền cửa sổ
        win_w = min(1280, max(960, int(screen_w * 0.94)))
        win_h = min(800, max(580, int(screen_h * 0.88)))
        pos_x = max(10, (screen_w - win_w) // 2)
        pos_y = max(10, (screen_h - win_h) // 2 - 20)

        self.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.minsize(940, 560)

    def _load_config(self) -> dict:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[LOI] Khong doc duoc file config: {e}")
        return {
            "hardware": {"camera_source": 1, "camera_resolution": [640, 480], "target_fps": 30, "camera_rotation": 0},
            "model": {"backend": "mediapipe", "unified_weights": "models/dms_unified.pt", "eye_classifier_weights": "models/eye_classifier.pt"},
            "preprocessing": {"dip_mode": "auto", "low_light_threshold": 85.0, "glare_threshold": 195.0, "smoothing_window": 5},
            "thresholds": {"ear_default": 0.22, "mar_default": 0.50, "pitch_default": -15.0, "yaw_default": 28.0, "sleep_time": 2.0, "nod_time": 1.5, "yawn_time": 2.0, "distraction_time": 2.0, "calibration_time": 5.0, "perclos_window": 60.0, "perclos_warn_threshold": 0.22, "perclos_threshold": 0.40},
        }

    def _start_system(self):
        try:
            self.status_banner.configure(
                text="● ĐANG KẾT NỐI CAMERA & NẠP AI...",
                fg_color="#854D0E",
                text_color="#FEF08A",
            )
            self.update_idletasks()

            # Khởi tạo mô-đun lõi
            self._init_core_modules()

            # Đồng bộ các thanh trượt theo giá trị thực tế của tracker
            if hasattr(self, "slider_ear") and self.tracker is not None:
                self.slider_ear.set(self.tracker.ear_thresh)
                self.slider_mar.set(self.tracker.mar_thresh)
                self.slider_sleep.set(self.tracker.sleep_time)
                self.slider_yawn.set(self.tracker.yawn_time)

            self.status_banner.configure(
                text="● TỈNH TÁO - AN TOÀN",
                fg_color="#064E3B",
                text_color="#D1FAE5",
            )
            self._add_log_entry("Khởi động Camera và mô hình AI MediaPipe + EyeCNN thành công!", "INFO")

            # Bắt đầu vòng lặp video
            self.after(20, self._video_loop)
        except Exception as e:
            self.status_banner.configure(
                text=f"[!] LỖI KHỞI ĐỘNG: {e}",
                fg_color="#991B1B",
                text_color="#FEE2E2",
            )
            self._add_log_entry(f"Lỗi khởi tạo hệ thống: {e}", "DANGER")

    def _init_core_modules(self):
        hw = self.config.get("hardware", {})
        md = self.config.get("model", {})
        pp = self.config.get("preprocessing", {})
        th = self.config.get("thresholds", {})

        # 1. Camera Stream
        cam_src = hw.get("camera_source", 1)
        cam_res = tuple(hw.get("camera_resolution", [640, 480]))
        cam_rot = int(hw.get("camera_rotation", 0))
        self.camera = WebcamStream(src=cam_src, resolution=cam_res, rotation=cam_rot).start()

        # 2. Xử lý ảnh số (DIP)
        self.img_processor = AdaptiveImageProcessor(
            mode=pp.get("dip_mode", "auto"),
            low_light_thresh=float(pp.get("low_light_threshold", 85.0)),
            glare_thresh=float(pp.get("glare_threshold", 195.0)),
        )

        # 3. AI Detector & Luồng bất đồng bộ
        backend = md.get("backend", "mediapipe")
        unified_w = os.path.join(self.root_dir, md.get("unified_weights", "models/dms_unified.pt"))
        eye_w = os.path.join(self.root_dir, md.get("eye_classifier_weights", "models/eye_classifier.pt"))
        self.detector = FaceLandmarkDetector(
            backend=backend,
            unified_weights=unified_w,
            eye_classifier_weights=eye_w,
            conf_threshold=float(md.get("conf_threshold", 0.5)),
            kpt_conf_threshold=float(md.get("kpt_conf_threshold", 0.3)),
            imgsz=int(md.get("imgsz", 416)),
        )
        self.ai_worker = AsyncDetectorWorker(self.detector).start()

        # 4. Bộ phân tích trạng thái tài xế
        self.tracker = DrowsinessTracker(thresholds_cfg=th.copy(), smoothing_window=int(pp.get("smoothing_window", 5)))

        # 5. Cảnh báo âm thanh
        audio_file = os.path.join(self.root_dir, hw.get("audio_file", "assets/alarm.wav"))
        self.alert = AlertSystem(method=hw.get("alert_method", "speaker"), sound_path=audio_file)

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ----------------- 1. THANH TIÊU ĐỀ TRÊN CÙNG (TOP BAR) -----------------
        self.top_bar = ctk.CTkFrame(self, height=52, corner_radius=8, fg_color="#181D26")
        self.top_bar.grid(row=0, column=0, padx=10, pady=(8, 4), sticky="nsew")
        self.top_bar.grid_columnconfigure(1, weight=1)

        # Logo / Tiêu đề ứng dụng
        self.app_title_lbl = ctk.CTkLabel(
            self.top_bar,
            text="● DMS DRIVER SAFETY",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#38BDF8",
        )
        self.app_title_lbl.grid(row=0, column=0, padx=14, pady=10, sticky="w")

        # Huy hiệu Trạng thái chính (Status Banner Pill)
        self.status_banner = ctk.CTkLabel(
            self.top_bar,
            text="● ĐANG KHỞI ĐỘNG HỆ THỐNG...",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#0F172A",
            text_color="#38BDF8",
            corner_radius=8,
            width=320,
            height=34,
        )
        self.status_banner.grid(row=0, column=1, padx=8, pady=8)

        # Thông số FPS & Hiệu năng
        self.perf_frame = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        self.perf_frame.grid(row=0, column=2, padx=12, pady=8, sticky="e")

        self.fps_badge = ctk.CTkLabel(
            self.perf_frame,
            text="FPS: 30.0",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#22C55E",
            fg_color="#0F291E",
            corner_radius=6,
            width=80,
            height=26,
        )
        self.fps_badge.pack(side="left", padx=3)

        self.ai_badge = ctk.CTkLabel(
            self.perf_frame,
            text="AI: 20ms",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38BDF8",
            fg_color="#0C2A44",
            corner_radius=6,
            width=80,
            height=26,
        )
        self.ai_badge.pack(side="left", padx=3)

        # ----------------- 2. KHÔNG GIAN LÀM VIỆC CHÍNH (MAIN WORKSPACE) -----------------
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)  # Cột Video mở rộng linh hoạt theo cửa sổ
        self.main_container.grid_columnconfigure(1, weight=0, minsize=350)  # Cột Sidebar cố định tối thiểu 350px
        self.main_container.grid_rowconfigure(0, weight=1)

        # ===== CỘT TRÁI: VIDEO & THANH ĐIỀU KHIỂN NHANH =====
        self.left_col = ctk.CTkFrame(self.main_container, fg_color="#181D26", corner_radius=10)
        self.left_col.grid(row=0, column=0, padx=(0, 8), pady=0, sticky="nsew")
        self.left_col.grid_rowconfigure(0, weight=1)
        self.left_col.grid_columnconfigure(0, weight=1)

        # Khung hiển thị Video
        self.video_frame = ctk.CTkFrame(self.left_col, fg_color="#0B0F17", corner_radius=8)
        self.video_frame.grid(row=0, column=0, padx=10, pady=8, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.video_frame, text="Đang mở luồng Video...", text_color="#64748B")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Thanh nút bấm điều khiển nhanh dưới Video (Action Bar)
        self.action_bar = ctk.CTkFrame(self.left_col, fg_color="transparent")
        self.action_bar.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="ew")

        self.btn_run = ctk.CTkButton(
            self.action_bar,
            text="|| TẠM DỪNG",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#0284C7",
            hover_color="#0369A1",
            width=100,
            height=30,
            command=self._toggle_running,
        )
        self.btn_run.pack(side="left", padx=(0, 4))

        self.btn_calib = ctk.CTkButton(
            self.action_bar,
            text="[CALIB 5s]",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#D97706",
            hover_color="#B45309",
            width=105,
            height=30,
            command=self._trigger_calibration,
        )
        self.btn_calib.pack(side="left", padx=4)

        self.btn_rot = ctk.CTkButton(
            self.action_bar,
            text="↺ XOAY 90°",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#334155",
            hover_color="#475569",
            width=95,
            height=30,
            command=self._rotate_camera,
        )
        self.btn_rot.pack(side="left", padx=4)

        self.btn_sound = ctk.CTkButton(
            self.action_bar,
            text="[ÂM: BẬT]",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#15803D",
            hover_color="#166534",
            width=95,
            height=30,
            command=self._toggle_sound,
        )
        self.btn_sound.pack(side="left", padx=4)

        self.btn_hud = ctk.CTkButton(
            self.action_bar,
            text="[HUD]",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#334155",
            hover_color="#475569",
            width=85,
            height=30,
            command=self._toggle_hud_overlay,
        )
        self.btn_hud.pack(side="left", padx=4)

        self.btn_snapshot = ctk.CTkButton(
            self.action_bar,
            text="[CHỤP ẢNH]",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#4F46E5",
            hover_color="#4338CA",
            width=95,
            height=30,
            command=self._take_snapshot,
        )
        self.btn_snapshot.pack(side="right")

        # Hộp Nhật ký Sự kiện (Event Log Box)
        self.log_container = ctk.CTkFrame(self.left_col, fg_color="#0F172A", corner_radius=8)
        self.log_container.grid(row=2, column=0, padx=10, pady=(0, 8), sticky="ew")

        self.log_header = ctk.CTkFrame(self.log_container, fg_color="transparent")
        self.log_header.pack(fill="x", padx=8, pady=(4, 2))

        ctk.CTkLabel(
            self.log_header,
            text="● Nhật ký Cảnh báo & Trạng thái Thời gian thực:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#94A3B8",
        ).pack(side="left")

        ctk.CTkButton(
            self.log_header,
            text="Xóa log",
            width=50,
            height=20,
            font=ctk.CTkFont(size=10),
            fg_color="#1E293B",
            hover_color="#334155",
            command=self._clear_log,
        ).pack(side="right")

        self.log_box = ctk.CTkTextbox(
            self.log_container,
            height=70,
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#E2E8F0",
            fg_color="#090D16",
        )
        self.log_box.pack(fill="x", padx=8, pady=(0, 6))
        self._add_log_entry("Hệ thống khởi động thành công. Đang tải Camera và mô hình AI...", "INFO")

        # ===== CỘT PHẢI: BẢNG ĐIỀU KHIỂN & CHỈ SỐ SINH TRẮC (SCROLLABLE SIDEBAR) =====
        self.right_col = ctk.CTkScrollableFrame(self.main_container, width=350, fg_color="#181D26", corner_radius=10)
        self.right_col.grid(row=0, column=1, padx=(0, 0), pady=0, sticky="nsew")

        self._build_settings_card()
        self._build_biometrics_card()
        self._build_sliders_card()

    def _build_settings_card(self):
        """Khối 1: Cấu hình Thiết bị & Mô hình AI."""
        card = ctk.CTkFrame(self.right_col, fg_color="#0F172A", corner_radius=8)
        card.pack(fill="x", padx=6, pady=(6, 4))

        ctk.CTkLabel(card, text="⚙ CẤU HÌNH CAMERA & AI", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=10, pady=(8, 6))

        # Chọn Camera
        cam_row = ctk.CTkFrame(card, fg_color="transparent")
        cam_row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(cam_row, text="Nguồn Cam:", width=80, anchor="w", font=ctk.CTkFont(size=11)).pack(side="left")

        self.cam_opt = ctk.CTkOptionMenu(
            cam_row,
            values=["Cam 1 (DroidCam)", "Cam 0 (Laptop)", "Nhập IP Cam..."],
            command=self._on_camera_selected,
            height=26,
            font=ctk.CTkFont(size=11),
        )
        self.cam_opt.pack(side="left", fill="x", expand=True, padx=(4, 0))
        cfg_cam = self.config.get("hardware", {}).get("camera_source", 1)
        if isinstance(cfg_cam, int) and cfg_cam == 0:
            self.cam_opt.set("Cam 0 (Laptop)")
        else:
            self.cam_opt.set("Cam 1 (DroidCam)")

        # Chọn Chế độ AI
        ai_row = ctk.CTkFrame(card, fg_color="transparent")
        ai_row.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(ai_row, text="Chế độ AI:", width=80, anchor="w", font=ctk.CTkFont(size=11)).pack(side="left")

        self.ai_seg = ctk.CTkSegmentedButton(
            ai_row,
            values=["MediaPipe", "YOLO", "So Sánh"],
            command=self._on_ai_mode_changed,
            height=26,
            font=ctk.CTkFont(size=11),
        )
        self.ai_seg.pack(side="left", fill="x", expand=True, padx=(4, 0))
        cfg_backend = self.config.get("model", {}).get("backend", "mediapipe").lower()
        if cfg_backend == "mediapipe":
            self.ai_seg.set("MediaPipe")
        elif cfg_backend == "unified":
            self.ai_seg.set("YOLO")
        else:
            self.ai_seg.set("So Sánh")

        # Chọn Tiền xử lý Ảnh số (DIP)
        dip_row = ctk.CTkFrame(card, fg_color="transparent")
        dip_row.pack(fill="x", padx=10, pady=(3, 8))
        ctk.CTkLabel(dip_row, text="Tiền xử lý:", width=80, anchor="w", font=ctk.CTkFont(size=11)).pack(side="left")

        self.dip_seg = ctk.CTkSegmentedButton(
            dip_row,
            values=["Tự Động", "CLAHE", "Tắt"],
            command=self._on_dip_mode_changed,
            height=26,
            font=ctk.CTkFont(size=11),
        )
        self.dip_seg.pack(side="left", fill="x", expand=True, padx=(4, 0))
        cfg_dip = self.config.get("preprocessing", {}).get("dip_mode", "auto").lower()
        if cfg_dip == "auto":
            self.dip_seg.set("Tự Động")
        elif cfg_dip == "clahe":
            self.dip_seg.set("CLAHE")
        else:
            self.dip_seg.set("Tắt")

    def _build_biometrics_card(self):
        """Khối 2: Chỉ số Sinh trắc & Trạng thái Thời gian thực."""
        card = ctk.CTkFrame(self.right_col, fg_color="#0F172A", corner_radius=8)
        card.pack(fill="x", padx=6, pady=4)

        ctk.CTkLabel(card, text="★ CHỈ SỐ SINH TRẮC THỜI GIAN THỰC", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=10, pady=(8, 6))

        # EAR (Mắt)
        self.ear_lbl = self._create_metric_row(card, "Độ mở mắt (EAR):", "0.28 (0.22)")
        self.ear_bar = ctk.CTkProgressBar(card, height=8, fg_color="#1E293B", progress_color="#22C55E")
        self.ear_bar.pack(fill="x", padx=10, pady=(0, 6))
        self.ear_bar.set(0.65)

        # Eye CNN (Xác suất mắt nhắm)
        self.eye_cnn_lbl = self._create_metric_row(card, "Mắt nhắm (EyeCNN):", "5%")
        self.eye_cnn_bar = ctk.CTkProgressBar(card, height=8, fg_color="#1E293B", progress_color="#22C55E")
        self.eye_cnn_bar.pack(fill="x", padx=10, pady=(0, 6))
        self.eye_cnn_bar.set(0.05)

        # MAR (Miệng)
        self.mar_lbl = self._create_metric_row(card, "Độ mở miệng (MAR):", "0.15 (0.50)")
        self.mar_bar = ctk.CTkProgressBar(card, height=8, fg_color="#1E293B", progress_color="#38BDF8")
        self.mar_bar.pack(fill="x", padx=10, pady=(0, 6))
        self.mar_bar.set(0.20)

        # PERCLOS
        self.perclos_lbl = self._create_metric_row(card, "Mệt mỏi (PERCLOS):", "0.0% (max 40%)")
        self.perclos_bar = ctk.CTkProgressBar(card, height=8, fg_color="#1E293B", progress_color="#22C55E")
        self.perclos_bar.pack(fill="x", padx=10, pady=(0, 6))
        self.perclos_bar.set(0.0)

        # Tư thế đầu & Thống kê
        info_row = ctk.CTkFrame(card, fg_color="#1E293B", corner_radius=6)
        info_row.pack(fill="x", padx=10, pady=(2, 8))

        self.head_pose_lbl = ctk.CTkLabel(info_row, text="> Tư thế đầu: Pitch: 0.0° | Yaw: 0.0°", font=ctk.CTkFont(size=11))
        self.head_pose_lbl.pack(anchor="w", padx=6, pady=(4, 1))

        self.stats_lbl = ctk.CTkLabel(info_row, text="> Chớp mắt: 0 | Ngáp: 0 | Micro-sleep: 0", font=ctk.CTkFont(size=11, weight="bold"), text_color="#CBD5E1")
        self.stats_lbl.pack(anchor="w", padx=6, pady=(1, 4))

    def _create_metric_row(self, parent, title: str, init_val: str):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(2, 1))
        ctk.CTkLabel(row, text=title, font=ctk.CTkFont(size=11), anchor="w").pack(side="left")
        val_lbl = ctk.CTkLabel(row, text=init_val, font=ctk.CTkFont(size=11, weight="bold"), text_color="#38BDF8", anchor="e")
        val_lbl.pack(side="right")
        return val_lbl

    def _build_sliders_card(self):
        """Khối 3: Tinh chỉnh Ngưỡng trực tiếp (Interactive Sliders)."""
        card = ctk.CTkFrame(self.right_col, fg_color="#0F172A", corner_radius=8)
        card.pack(fill="x", padx=6, pady=4)

        ctk.CTkLabel(card, text="⚙ TINH CHỈNH ĐỘ NHẠY (SLIDERS)", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=10, pady=(8, 6))

        th = self.config.get("thresholds", {})
        init_ear = float(th.get("ear_default", 0.22))
        init_mar = float(th.get("mar_default", 0.50))
        init_sleep = float(th.get("sleep_time", 2.0))
        init_yawn = float(th.get("yawn_time", 2.0))

        # Slider EAR
        self.slider_ear_val_lbl = ctk.CTkLabel(card, text=f"Ngưỡng EAR Nhắm mắt: {init_ear:.2f}", font=ctk.CTkFont(size=11))
        self.slider_ear_val_lbl.pack(anchor="w", padx=10, pady=(2, 0))
        self.slider_ear = ctk.CTkSlider(card, from_=0.15, to=0.32, number_of_steps=34, height=14, command=self._on_ear_slider)
        self.slider_ear.set(init_ear)
        self.slider_ear.pack(fill="x", padx=10, pady=(0, 4))

        # Slider MAR
        self.slider_mar_val_lbl = ctk.CTkLabel(card, text=f"Ngưỡng MAR Ngáp: {init_mar:.2f}", font=ctk.CTkFont(size=11))
        self.slider_mar_val_lbl.pack(anchor="w", padx=10, pady=(2, 0))
        self.slider_mar = ctk.CTkSlider(card, from_=0.35, to=0.70, number_of_steps=35, height=14, command=self._on_mar_slider)
        self.slider_mar.set(init_mar)
        self.slider_mar.pack(fill="x", padx=10, pady=(0, 4))

        # Slider Sleep Time
        self.slider_sleep_val_lbl = ctk.CTkLabel(card, text=f"Thời gian báo ngủ gật: {init_sleep:.1f}s", font=ctk.CTkFont(size=11))
        self.slider_sleep_val_lbl.pack(anchor="w", padx=10, pady=(2, 0))
        self.slider_sleep = ctk.CTkSlider(card, from_=1.0, to=3.5, number_of_steps=25, height=14, command=self._on_sleep_slider)
        self.slider_sleep.set(init_sleep)
        self.slider_sleep.pack(fill="x", padx=10, pady=(0, 4))

        # Slider Yawn Time
        self.slider_yawn_val_lbl = ctk.CTkLabel(card, text=f"Thời gian xác nhận ngáp: {init_yawn:.1f}s", font=ctk.CTkFont(size=11))
        self.slider_yawn_val_lbl.pack(anchor="w", padx=10, pady=(2, 0))
        self.slider_yawn = ctk.CTkSlider(card, from_=1.0, to=3.5, number_of_steps=25, height=14, command=self._on_yawn_slider)
        self.slider_yawn.set(init_yawn)
        self.slider_yawn.pack(fill="x", padx=10, pady=(0, 8))

        # Nút Lưu cấu hình & Khôi phục mặc định
        btn_box = ctk.CTkFrame(card, fg_color="transparent")
        btn_box.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkButton(
            btn_box,
            text="[LƯU CẤU HÌNH]",
            height=26,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#047857",
            hover_color="#065F46",
            command=self._save_settings_to_file,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        ctk.CTkButton(
            btn_box,
            text="↺ Mặc định",
            height=26,
            width=75,
            font=ctk.CTkFont(size=11),
            fg_color="#334155",
            hover_color="#475569",
            command=self._reset_sliders_default,
        ).pack(side="right")

    # ----------------- CÁC HÀM XỬ LÝ SỰ KIỆN GIAO DIỆN -----------------

    def _toggle_running(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.btn_run.configure(text="⏸ TẠM DỪNG", fg_color="#0284C7", hover_color="#0369A1")
            self._add_log_entry("Đã tiếp tục giám sát.", "INFO")
            self.after(20, self._video_loop)
        else:
            self.btn_run.configure(text="▶ TIẾP TỤC", fg_color="#16A34A", hover_color="#15803D")
            if hasattr(self, "alert") and self.alert is not None:
                self.alert.stop()
            self._add_log_entry("Đã tạm dừng giám sát.", "WARN")

    def _trigger_calibration(self):
        if hasattr(self, "alert") and self.alert is not None:
            self.alert.stop()
        if hasattr(self, "tracker") and self.tracker is not None:
            self.tracker.reset_calibration()
            self._add_log_entry("Bắt đầu chu trình hiệu chuẩn 5 giây...", "CALIB")

    def _rotate_camera(self):
        if hasattr(self, "camera") and self.camera is not None:
            new_rot = self.camera.cycle_rotation()
            self._add_log_entry(f"Đã xoay khung hình camera: {new_rot}°", "INFO")

    def _toggle_sound(self):
        self.is_muted = not self.is_muted
        if self.is_muted:
            if hasattr(self, "alert") and self.alert is not None:
                self.alert.stop()
            self.btn_sound.configure(text="🔇 ÂM: TẮT", fg_color="#DC2626", hover_color="#B91C1C")
            self._add_log_entry("Đã tắt âm thanh cảnh báo.", "WARN")
        else:
            self.btn_sound.configure(text="🔊 ÂM: BẬT", fg_color="#15803D", hover_color="#166534")
            self._add_log_entry("Đã bật lại âm thanh cảnh báo.", "INFO")

    def _toggle_hud_overlay(self):
        self.show_hud_text = not self.show_hud_text
        if self.show_hud_text:
            self.btn_hud.configure(fg_color="#0284C7", hover_color="#0369A1")
            self._add_log_entry("Đã BẬT lớp phủ chữ HUD trên video.", "INFO")
        else:
            self.btn_hud.configure(fg_color="#334155", hover_color="#475569")
            self._add_log_entry("Đã TẮT lớp phủ chữ HUD (khung hình video sạch sẽ).", "INFO")

    def _take_snapshot(self):
        if not hasattr(self, "camera") or self.camera is None:
            return
        success, frame = self.camera.read()
        if success and frame is not None:
            snap_dir = os.path.join(self.root_dir, "snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            fname = f"dms_snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            fpath = os.path.join(snap_dir, fname)
            cv2.imwrite(fpath, frame)
            self._add_log_entry(f"Đã lưu ảnh chụp tại: snapshots/{fname}", "INFO")

    def _clear_log(self):
        self.log_box.delete("1.0", "end")

    def _add_log_entry(self, message: str, level: str = "INFO"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        prefix = f"[{ts}] [{level}] "
        self.log_box.insert("end", prefix + message + "\n")
        try:
            line_count = int(self.log_box.index("end-1c").split(".")[0])
            if line_count > 120:
                self.log_box.delete("1.0", f"{line_count - 100}.0")
        except Exception:
            pass
        self.log_box.see("end")

    def _on_camera_selected(self, choice: str):
        if not hasattr(self, "camera") or self.camera is None:
            return
        if "Cam 0" in choice:
            self.camera.switch_source(0)
            if hasattr(self, "tracker") and self.tracker is not None:
                self.tracker.reset_calibration()
            self._add_log_entry("Đã chuyển sang Camera Laptop (Index 0).", "INFO")
        elif "Cam 1" in choice:
            self.camera.switch_source(1)
            if hasattr(self, "tracker") and self.tracker is not None:
                self.tracker.reset_calibration()
            self._add_log_entry("Đã chuyển sang DroidCam Virtual Camera (Index 1).", "INFO")
        elif "Nhập IP" in choice:
            dialog = ctk.CTkInputDialog(text="Nhập địa chỉ IP DroidCam (VD: 192.168.1.82:4747):", title="Kết nối DroidCam IP")
            ip_val = dialog.get_input()
            if ip_val and ip_val.strip():
                ip_str = ip_val.strip()
                src_url = ip_str if ip_str.startswith("http") else f"http://{ip_str}/video"
                self.camera.switch_source(src_url)
                if hasattr(self, "tracker") and self.tracker is not None:
                    self.tracker.reset_calibration()
                self._add_log_entry(f"Đang kết nối camera qua mạng: {src_url}", "INFO")

    def _on_ai_mode_changed(self, choice: str):
        if not hasattr(self, "detector") or self.detector is None:
            return
        mode_map = {
            "MediaPipe": "mediapipe",
            "MediaPipe 3D": "mediapipe",
            "YOLO": "unified",
            "Unified YOLO": "unified",
            "So Sánh": "compare",
        }
        target_mode = mode_map.get(choice, "mediapipe")
        try:
            self.detector.set_backend(target_mode)
            self._add_log_entry(f"Đã chuyển sang chế độ AI: {choice.upper()}", "INFO")
        except Exception as e:
            self._add_log_entry(f"Lỗi chuyển chế độ AI: {e}", "WARN")

    def _on_dip_mode_changed(self, choice: str):
        if not hasattr(self, "img_processor") or self.img_processor is None:
            return
        dip_map = {
            "Tự Động": "auto",
            "Tự Động (Auto)": "auto",
            "CLAHE": "clahe",
            "Tắt": "off",
            "Tắt (Off)": "off",
        }
        target_dip = dip_map.get(choice, "auto")
        self.img_processor.mode = target_dip
        self._add_log_entry(f"Đã chuyển chế độ tiền xử lý ảnh số sang: {choice}", "INFO")

    def _on_ear_slider(self, val: float):
        if hasattr(self, "tracker") and self.tracker is not None:
            self.tracker.ear_thresh = float(val)
        self.slider_ear_val_lbl.configure(text=f"Ngưỡng EAR Nhắm mắt: {float(val):.2f}")

    def _on_mar_slider(self, val: float):
        if hasattr(self, "tracker") and self.tracker is not None:
            self.tracker.mar_thresh = float(val)
        self.slider_mar_val_lbl.configure(text=f"Ngưỡng MAR Ngáp: {float(val):.2f}")

    def _on_sleep_slider(self, val: float):
        if hasattr(self, "tracker") and self.tracker is not None:
            self.tracker.sleep_time = float(val)
        self.slider_sleep_val_lbl.configure(text=f"Thời gian báo ngủ gật: {float(val):.1f}s")

    def _on_yawn_slider(self, val: float):
        if hasattr(self, "tracker") and self.tracker is not None:
            self.tracker.yawn_time = float(val)
        self.slider_yawn_val_lbl.configure(text=f"Thời gian xác nhận ngáp: {float(val):.1f}s")

    def _save_settings_to_file(self):
        th = self.config.setdefault("thresholds", {})
        th["ear_default"] = round(float(self.slider_ear.get()), 2)
        th["mar_default"] = round(float(self.slider_mar.get()), 2)
        th["sleep_time"] = round(float(self.slider_sleep.get()), 1)
        th["yawn_time"] = round(float(self.slider_yawn.get()), 1)
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
            self._add_log_entry("Đã lưu các ngưỡng nhạy vào config.json thành công!", "INFO")
        except Exception as e:
            self._add_log_entry(f"Lỗi khi lưu config: {e}", "WARN")

    def _reset_sliders_default(self):
        self.slider_ear.set(0.22)
        self._on_ear_slider(0.22)
        self.slider_mar.set(0.50)
        self._on_mar_slider(0.50)
        self.slider_sleep.set(2.0)
        self._on_sleep_slider(2.0)
        self.slider_yawn.set(2.0)
        self._on_yawn_slider(2.0)
        self._add_log_entry("Đã khôi phục các ngưỡng về mức tiêu chuẩn.", "INFO")

    # ----------------- VÒNG LẶP XỬ LÝ VIDEO & CẬP NHẬT GIAO DIỆN -----------------

    def _video_loop(self):
        if not self.is_running:
            return

        try:
            if not self.winfo_exists():
                return
        except Exception:
            return

        if self.camera is None or self.ai_worker is None or self.tracker is None:
            self.after(50, self._video_loop)
            return

        loop_start = time.perf_counter()

        success, frame = self.camera.read()
        if not success or frame is None:
            self.after(20, self._video_loop)
            return

        # 1. Tiền xử lý ảnh số vùng mặt
        proc_frame, dip_info = self.img_processor.process(frame, face_bbox=self.last_face_bbox)

        # 2. Suy luận AI bất đồng bộ
        self.ai_worker.submit_frame(proc_frame)
        detection = self.ai_worker.get_result()
        if detection.detected and detection.bbox is not None:
            self.last_face_bbox = detection.bbox

        # 3. Phân tích trạng thái hành vi tài xế
        status = self.tracker.update(detection)

        # 4. Quản lý âm thanh cảnh báo
        if not self.is_muted and self.alert is not None:
            self.alert.update_level(status.alert_level)
        elif self.alert is not None:
            self.alert.stop()

        # 5. Ghi log khi trạng thái cảnh báo thay đổi
        if status.alert_level != self.last_status_level:
            if status.alert_level == 2:
                self._add_log_entry(f"NGUY HIỂM: {status.status_text} (EAR: {status.ear:.2f}, Sleep: {status.sleep_elapsed:.1f}s)", "DANGER")
            elif status.alert_level == 1:
                self._add_log_entry(f"CHÚ Ý: {status.status_text}", "WARN")
            elif status.alert_level == 0 and self.last_status_level > 0:
                self._add_log_entry("Tài xế đã an toàn / Tỉnh táo trở lại.", "SAFE")
            self.last_status_level = status.alert_level

        # 6. Tính toán FPS
        now = time.perf_counter()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0:
            self.fps_history.append(1.0 / dt)
        cur_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 30.0

        # 7. Cập nhật các bảng đo và chỉ số trên giao diện (Điều tiết 8Hz để giải phóng CPU)
        urgent = (
            status.alert_level != self._last_ui_alert_level
            or status.is_calibrating != self._last_ui_calib_state
            or status.status_text != self._last_ui_status_text
        )
        should_full_update = (now - self._last_metrics_update) >= 0.125
        if urgent or should_full_update:
            self._update_ui_metrics(detection, status, cur_fps, full_update=should_full_update)
            if should_full_update:
                self._last_metrics_update = now
            self._last_ui_alert_level = status.alert_level
            self._last_ui_calib_state = status.is_calibrating
            self._last_ui_status_text = status.status_text

        # 8. Vẽ lớp phủ HUD và hiển thị lên khung video
        display_frame = DMSVisualizer.draw_hud(
            proc_frame,
            detection,
            status,
            cur_fps,
            dip_info,
            cam_info=f"Cam: {self.camera.src} | AI: {self.ai_worker.ai_fps:.0f}fps",
            show_text_overlay=self.show_hud_text,
        )

        self._render_frame_to_label(display_frame)

        # Điều nhịp thời gian để giữ mượt mà chuẩn 30 FPS (~33ms/frame)
        elapsed = time.perf_counter() - loop_start
        remaining = (1.0 / 30.0) - elapsed
        delay_ms = max(1, int(remaining * 1000)) if remaining > 0.002 else 1
        self.after(delay_ms, self._video_loop)

    def _update_ui_metrics(self, detection, status: DriverStatus, fps: float, full_update: bool = True):
        # 1. Luôn cập nhật tức thì trạng thái cảnh báo khẩn cấp (0ms delay)
        if status.is_calibrating:
            self.status_banner.configure(
                text=f"↺ ĐANG HIỆU CHUẨN... ({status.calib_time_left}s)",
                fg_color="#854D0E",
                text_color="#FEF08A",
            )
        elif status.alert_level >= 2:
            self.status_banner.configure(
                text=f"■ NGUY HIỂM: {status.status_text}",
                fg_color="#991B1B",
                text_color="#FEE2E2",
            )
        elif status.alert_level == 1:
            self.status_banner.configure(
                text=f"▲ CHÚ Ý: {status.status_text}",
                fg_color="#9A3412",
                text_color="#FFEDD5",
            )
        else:
            self.status_banner.configure(
                text="● TỈNH TÁO - AN TOÀN",
                fg_color="#064E3B",
                text_color="#D1FAE5",
            )

        # Bỏ qua 15 widget số liệu nếu chưa đến chu kỳ 125ms (~8 Hz) để giữ video 30 FPS mượt mà
        if not full_update:
            return

        self.fps_badge.configure(text=f"FPS: {fps:.1f}")
        ai_fps = self.ai_worker.ai_fps if self.ai_worker else 0.0
        self.ai_badge.configure(text=f"AI: {ai_fps:.0f}fps")

        # Thanh đo EAR
        ear_color = "#EF4444" if (status.ear > 0 and status.ear < status.ear_thresh) else "#22C55E"
        self.ear_lbl.configure(text=f"{status.ear:.2f} ({status.ear_thresh:.2f})", text_color=ear_color)
        norm_ear = min(1.0, max(0.0, status.ear / 0.40))
        self.ear_bar.configure(progress_color=ear_color)
        self.ear_bar.set(norm_ear)

        # Thanh đo Eye CNN Sleepy
        cnn_pct = status.eye_closed_prob * 100.0
        cnn_color = "#EF4444" if status.eye_closed_prob >= 0.58 else ("#F59E0B" if status.eye_closed_prob >= 0.30 else "#22C55E")
        self.eye_cnn_lbl.configure(text=f"{cnn_pct:.0f}%", text_color=cnn_color)
        self.eye_cnn_bar.configure(progress_color=cnn_color)
        self.eye_cnn_bar.set(min(1.0, max(0.0, status.eye_closed_prob)))

        # Thanh đo MAR
        mar_color = "#EF4444" if status.mar > status.mar_thresh else "#38BDF8"
        self.mar_lbl.configure(text=f"{status.mar:.2f} ({status.mar_thresh:.2f})", text_color=mar_color)
        norm_mar = min(1.0, max(0.0, status.mar / 0.80))
        self.mar_bar.configure(progress_color=mar_color)
        self.mar_bar.set(norm_mar)

        # Thanh đo PERCLOS
        perclos_pct = status.perclos * 100.0
        p_thresh = getattr(status, "perclos_thresh", 0.40)
        p_warn_thresh = getattr(status, "perclos_warn_thresh", 0.22)
        p_color = "#EF4444" if status.perclos >= p_thresh else ("#F59E0B" if status.perclos >= p_warn_thresh else "#22C55E")
        self.perclos_lbl.configure(text=f"{perclos_pct:.1f}% (max {p_thresh*100:.0f}%)", text_color=p_color)
        self.perclos_bar.configure(progress_color=p_color)
        self.perclos_bar.set(min(1.0, max(0.0, status.perclos / 0.50)))

        # Tư thế đầu & Thống kê
        self.head_pose_lbl.configure(text=f"> Tư thế đầu: Pitch: {status.pitch:+.1f}° | Yaw: {status.yaw:+.1f}°")
        self.stats_lbl.configure(text=f"> Chớp mắt: {status.blink_count} | Ngáp: {status.yawn_count} | Micro-sleep: {status.microsleep_count}")

    def _render_frame_to_label(self, frame_bgr: np.ndarray):
        """Render khung hình trực tiếp bằng ImageTk.PhotoImage (>60 FPS, không nghẽn CPU)."""
        try:
            if not self.winfo_exists():
                return
            lbl_w = max(320, self.video_frame.winfo_width())
            lbl_h = max(240, self.video_frame.winfo_height())

            h, w = frame_bgr.shape[:2]
            scale = min(lbl_w / w, lbl_h / h)
            new_w, new_h = max(32, int(w * scale)), max(32, int(h * scale))

            if new_w > 10 and new_h > 10:
                resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tk_img = ImageTk.PhotoImage(pil_img)
                self._current_photo_img = tk_img  # Lưu reference chống GC thu hồi bộ nhớ

                # Cập nhật trực tiếp lên nhãn hiển thị
                if not self._video_text_cleared:
                    self.video_label._label.configure(image=tk_img, text="")
                    self._video_text_cleared = True
                else:
                    self.video_label._label.configure(image=tk_img)
        except Exception:
            pass

    def on_close(self):
        print("[App] Dang dung he thong va giai phong tai nguyen...")
        self.is_running = False
        try:
            if hasattr(self, "alert") and self.alert is not None:
                self.alert.stop()
            if hasattr(self, "ai_worker") and self.ai_worker is not None:
                self.ai_worker.stop()
            if hasattr(self, "camera") and self.camera is not None:
                self.camera.stop()
        except Exception:
            pass
        try:
            self.withdraw()
            self.quit()
        except Exception:
            pass


def launch_gui(config_path: str = "config.json"):
    app = DMSApp(config_path=config_path)
    try:
        app.mainloop()
    except Exception:
        pass
    try:
        app.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    launch_gui()
