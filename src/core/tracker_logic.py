import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .detector import DetectionResult
from .image_processing import OneEuroFilter1D


@dataclass
class DriverStatus:
    """Trạng thái chi tiết của tài xế tại mỗi khung hình."""
    is_calibrating: bool = False
    calib_time_left: int = 0
    status_text: str = "TINH TAO"
    status_color: Tuple[int, int, int] = (0, 255, 0)
    alert_level: int = 0                    # 0: SAFE, 1: WARNING (Nhắc nhở), 2: DANGER (Báo động khẩn cấp)
    trigger_alarm: bool = False             # True khi alert_level == 2 (tương thích ngược)
    danger_progress: float = 0.0            # Tiến trình tiến tới ngưỡng báo động [0.0 -> 1.0]
    is_yawning: bool = False
    yawn_count: int = 0
    blink_count: int = 0
    microsleep_count: int = 0
    ear: float = 0.0
    mar: float = 0.0
    eye_closed_prob: float = 0.0            # Xac suat nham mat Tang 2 (CNN MRL Dataset) [0.0 -> 1.0]
    pitch: float = 0.0
    yaw: float = 0.0
    ear_thresh: float = 0.22
    mar_thresh: float = 0.50
    pitch_thresh: float = -15.0
    yaw_thresh: float = 28.0
    perclos: float = 0.0
    sleep_elapsed: float = 0.0
    yawn_elapsed: float = 0.0
    nod_elapsed: float = 0.0
    distract_elapsed: float = 0.0


class DrowsinessTracker:
    """
    Hệ thống theo dõi và phân tích hành vi tài xế thời gian thực (FPS-independent):
    - Lọc nhiễu tín hiệu bằng One-Euro Filter 1D.
    - Hợp nhất 2 tầng (Stage-1 EAR hình học + Stage-2 CNN phân loại vùng mắt từ MRL Eye Dataset).
    - Bù trừ góc nghiêng đầu cho chỉ số EAR (Head-Pose Pitch Compensation) chống báo giả khi nhìn bảng đồng hồ.
    - Bộ phân biệt Nói chuyện vs. Ngáp (Talking vs. Yawning Discriminator) qua độ biến thiên MAR.
    - Tự động hiệu chuẩn (Auto-Calibration) cá nhân hóa theo mắt, miệng và góc nhìn thẳng của từng tài xế.
    """

    def __init__(self, thresholds_cfg: dict, smoothing_window: int = 5):
        self.ear_thresh = float(thresholds_cfg.get("ear_default", 0.22))
        self.mar_thresh = float(thresholds_cfg.get("mar_default", 0.50))
        self.pitch_thresh = float(thresholds_cfg.get("pitch_default", -15.0))
        self.yaw_thresh = float(thresholds_cfg.get("yaw_default", 28.0))
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0

        self.sleep_time = float(thresholds_cfg.get("sleep_time", 2.0))
        self.nod_time = float(thresholds_cfg.get("nod_time", 1.5))
        self.yawn_time = float(thresholds_cfg.get("yawn_time", 2.0))
        self.distraction_time = float(thresholds_cfg.get("distraction_time", 2.0))
        self.calib_duration = float(thresholds_cfg.get("calibration_time", 5.0))

        self.perclos_window = float(thresholds_cfg.get("perclos_window", 60.0))
        self.perclos_warn_thresh = float(thresholds_cfg.get("perclos_warn_threshold", 0.22))
        self.perclos_thresh = float(thresholds_cfg.get("perclos_threshold", 0.40))

        # Bộ lọc thích nghi One-Euro Filter cho từng tín hiệu sinh trắc
        self.ear_filter = OneEuroFilter1D(min_cutoff=1.4, beta=0.25)
        self.mar_filter = OneEuroFilter1D(min_cutoff=1.2, beta=0.20)
        self.eye_prob_filter = OneEuroFilter1D(min_cutoff=1.5, beta=0.30)
        self.pitch_filter = OneEuroFilter1D(min_cutoff=1.0, beta=0.15)
        self.yaw_filter = OneEuroFilter1D(min_cutoff=1.0, beta=0.15)

        # Lưu lịch sử (timestamp, is_eye_closed) để tính chỉ số PERCLOS
        self.eye_closure_log = deque()
        # Lưu lịch sử MAR ngắn hạn (30 mẫu gần nhất) để phân biệt Nói chuyện (dao động mạnh) vs Ngáp (mở ổn định)
        self.mar_recent_window = deque(maxlen=30)

        self.is_calibrating = self.calib_duration > 0
        self.calib_start_time = None
        self.calib_ear_data = []
        self.calib_mar_data = []
        self.calib_pitch_data = []
        self.calib_yaw_data = []

        self.sleep_start = None
        self.nod_start = None
        self.yawn_start = None
        self.distract_start = None

        self._yawn_counted_in_current_bout = False
        self.yawn_count = 0
        self.blink_count = 0
        self.microsleep_count = 0

    def reset_calibration(self):
        """Khởi động lại quá trình hiệu chuẩn cá nhân hóa (khi bấm phím 'C')."""
        self.is_calibrating = True
        self.calib_start_time = None
        self.calib_ear_data.clear()
        self.calib_mar_data.clear()
        self.calib_pitch_data.clear()
        self.calib_yaw_data.clear()
        self.mar_recent_window.clear()
        self.ear_filter.reset()
        self.mar_filter.reset()
        self.eye_prob_filter.reset()
        self.pitch_filter.reset()
        self.yaw_filter.reset()
        self.sleep_start = None
        self.nod_start = None
        self.yawn_start = None
        self.distract_start = None
        print("[Calibration] Bat dau hieu chuan lai tu the va mat tai xe...")

    def update(self, detection: DetectionResult) -> DriverStatus:
        """Cập nhật trạng thái dựa trên kết quả nhận diện của khung hình hiện tại."""
        now = time.time()

        if not detection.detected:
            return DriverStatus(
                is_calibrating=self.is_calibrating,
                status_text="KHONG THAY KHUON MAT!",
                status_color=(0, 165, 255),
                alert_level=0,
                trigger_alarm=False,
                yawn_count=self.yawn_count,
                blink_count=self.blink_count,
                microsleep_count=self.microsleep_count,
                ear_thresh=self.ear_thresh,
                mar_thresh=self.mar_thresh,
                pitch_thresh=self.pitch_thresh,
                yaw_thresh=self.yaw_thresh,
            )

        # 1. Lọc nhiễu tín hiệu bằng One-Euro Filter 1D
        raw_ear = self.ear_filter.filter(detection.ear, now) if detection.ear > 0 else 0.0
        mar = self.mar_filter.filter(detection.mar, now) if detection.mar > 0 else 0.0
        eye_closed_prob = (
            self.eye_prob_filter.filter(detection.eye_closed_prob, now)
            if detection.eye_closed_prob is not None
            else -1.0
        )
        pitch = self.pitch_filter.filter(detection.pitch, now) if detection.pitch is not None else 0.0
        yaw = self.yaw_filter.filter(detection.yaw, now) if detection.yaw is not None else 0.0

        # 2. Pha Hiệu Chuẩn (Calibration)
        if self.is_calibrating:
            if self.calib_start_time is None:
                self.calib_start_time = now
            elapsed = now - self.calib_start_time
            time_left = max(0, int(round(self.calib_duration - elapsed)))
            calib_progress = min(1.0, elapsed / max(0.1, self.calib_duration))

            if elapsed < self.calib_duration:
                # Chỉ thu thập mẫu EAR mắt mở khi mạng CNN Tầng 2 xác nhận mắt đang mở (hoặc khi CNN không bật)
                if raw_ear > 0.12 and (eye_closed_prob < 0 or eye_closed_prob < 0.50):
                    self.calib_ear_data.append(raw_ear)
                if mar > 0.0:
                    self.calib_mar_data.append(mar)
                if detection.pitch is not None:
                    self.calib_pitch_data.append(pitch)
                if detection.yaw is not None:
                    self.calib_yaw_data.append(yaw)

                return DriverStatus(
                    is_calibrating=True,
                    calib_time_left=time_left,
                    status_text=f"DANG HIEU CHUAN... {time_left}s",
                    status_color=(0, 255, 255),
                    alert_level=0,
                    trigger_alarm=False,
                    danger_progress=calib_progress,
                    ear=raw_ear,
                    mar=mar,
                    eye_closed_prob=max(0.0, eye_closed_prob),
                    pitch=pitch,
                    yaw=yaw,
                    ear_thresh=self.ear_thresh,
                    mar_thresh=self.mar_thresh,
                    pitch_thresh=self.pitch_thresh,
                    yaw_thresh=self.yaw_thresh,
                )
            else:
                if self.calib_ear_data:
                    # Loại bỏ 15% giá trị thấp nhất (phòng trường hợp tài xế chớp mắt trong lúc hiệu chuẩn)
                    sorted_ear = sorted(self.calib_ear_data)
                    trim_idx = int(len(sorted_ear) * 0.15)
                    open_ear_samples = sorted_ear[trim_idx:] if trim_idx < len(sorted_ear) else sorted_ear
                    mean_open_ear = sum(open_ear_samples) / len(open_ear_samples)
                    self.ear_thresh = max(0.15, min(0.28, mean_open_ear * 0.76))
                if self.calib_mar_data:
                    self.mar_thresh = max(0.42, (sum(self.calib_mar_data) / len(self.calib_mar_data)) + 0.28)
                if self.calib_pitch_data:
                    self.baseline_pitch = sum(self.calib_pitch_data) / len(self.calib_pitch_data)
                    self.pitch_thresh = self.baseline_pitch - 12.0
                if self.calib_yaw_data:
                    self.baseline_yaw = sum(self.calib_yaw_data) / len(self.calib_yaw_data)

                self.is_calibrating = False
                print(
                    f"[Hieu chuan xong] EAR_THRESH={self.ear_thresh:.2f}, "
                    f"MAR_THRESH={self.mar_thresh:.2f}, PITCH_THRESH={self.pitch_thresh:.1f}, "
                    f"BASELINE_YAW={self.baseline_yaw:.1f}"
                )

        # 3. Bù trừ góc nghiêng đầu cho EAR (Head-Pose Pitch Compensation)
        # Khi tài xế hơi nhìn xuống đồng hồ (nhưng chưa tới mức gục đầu), bù hệ số cos(delta_pitch)
        delta_pitch = pitch - self.baseline_pitch
        if -12.0 < delta_pitch < -2.0:
            cos_factor = max(0.82, math.cos(math.radians(abs(delta_pitch) * 1.8)))
            ear = raw_ear / cos_factor
        else:
            ear = raw_ear

        # 4. Hợp nhất quyết định Nhắm Mắt 2 Tầng (Stage-1 EAR + Stage-2 EyeStateNet CNN):
        # - Nếu không có CNN Tầng 2: dùng thuần ngưỡng EAR.
        # - Nếu có CNN Tầng 2 (eye_closed_prob >= 0):
        #   + Bắt nhạy mắt nhắm thực sự ngay cả khi YOLO bị hở giả (pseudo-label gap):
        #     khi CNN tự tin mắt nhắm >= 78%, hoặc CNN >= 60% đồng thời EAR hơi thấp (< ear_thresh * 1.18).
        #   + Chống báo giả khi nhìn xuống hoặc nheo mắt nhẹ: nếu EAR hơi thấp hơn ngưỡng
        #     nhưng CNN nhìn rõ con ngươi đang mở (eye_closed_prob < 0.28 và ear > ear_thresh * 0.72) -> Mở mắt.
        ear_says_closed = (ear > 0) and (ear < self.ear_thresh)
        if eye_closed_prob >= 0.0:
            if eye_closed_prob >= 0.78:
                is_eye_closed = True
            elif eye_closed_prob >= 0.58 and (ear > 0 and ear < self.ear_thresh * 1.18):
                is_eye_closed = True
            elif ear_says_closed and eye_closed_prob < 0.28 and ear > self.ear_thresh * 0.72:
                is_eye_closed = False
            else:
                is_eye_closed = ear_says_closed
        else:
            is_eye_closed = ear_says_closed

        # 5. Cập nhật lịch sử PERCLOS (cửa sổ trượt perclos_window giây)
        self.eye_closure_log.append((now, 1 if is_eye_closed else 0))
        while self.eye_closure_log and (now - self.eye_closure_log[0][0]) > self.perclos_window:
            self.eye_closure_log.popleft()
        perclos = (
            sum(flag for _, flag in self.eye_closure_log) / len(self.eye_closure_log)
            if len(self.eye_closure_log) >= 15
            else 0.0
        )

        # 5. Đánh giá các trạng thái cảnh báo theo thời gian thực (giây)
        status_text = "TINH TAO"
        status_color = (0, 255, 0)
        alert_level = 0
        is_yawning = False
        sleep_elapsed = 0.0
        nod_elapsed = 0.0
        yawn_elapsed = 0.0
        distract_elapsed = 0.0

        # A. Kiểm tra Nhắm mắt / Chớp mắt / Micro-sleep / Ngủ gật
        if is_eye_closed:
            if self.sleep_start is None:
                self.sleep_start = now
            sleep_elapsed = now - self.sleep_start
            if sleep_elapsed >= self.sleep_time:
                status_text = "CANH BAO: NGU GAT!"
                status_color = (0, 0, 255)
                alert_level = 2
            elif sleep_elapsed >= 0.75:
                status_text = "CHU Y: CHOP MAT LAU (MICRO-SLEEP)!"
                status_color = (0, 140, 255)
                alert_level = max(alert_level, 1)
        else:
            if self.sleep_start is not None:
                bout_duration = now - self.sleep_start
                if 0.08 <= bout_duration <= 0.45:
                    self.blink_count += 1
                elif 0.75 <= bout_duration < self.sleep_time:
                    self.microsleep_count += 1
            self.sleep_start = None

        # B. Kiểm tra Gục đầu (Pitch)
        if detection.pitch is not None and pitch < self.pitch_thresh:
            if self.nod_start is None:
                self.nod_start = now
            nod_elapsed = now - self.nod_start
            if nod_elapsed >= self.nod_time and alert_level < 2:
                status_text = "CANH BAO: GUC DAU!"
                status_color = (0, 0, 255)
                alert_level = 2
        else:
            self.nod_start = None

        # C. Kiểm tra Mất tập trung / Quay đầu nhìn sang hai bên (Yaw)
        if detection.yaw is not None and abs(yaw - self.baseline_yaw) > self.yaw_thresh:
            if self.distract_start is None:
                self.distract_start = now
            distract_elapsed = now - self.distract_start
            if distract_elapsed >= self.distraction_time and alert_level < 2:
                status_text = "CHU Y: MAT TAP TRUNG (QUAY DAU)!"
                status_color = (0, 165, 255)
                alert_level = max(alert_level, 1)
        else:
            self.distract_start = None

        # D. Kiểm tra Ngáp vs Nói chuyện (Talking vs Yawning Discriminator)
        self.mar_recent_window.append(mar)
        if mar > self.mar_thresh:
            if self.yawn_start is None:
                self.yawn_start = now
                self._yawn_counted_in_current_bout = False
            yawn_elapsed = now - self.yawn_start

            # Phân biệt nói chuyện: khi nói chuyện MAR dao động lên xuống liên tục (std cao),
            # khi ngáp miệng mở lớn duy trì ổn định
            mar_std = float(np.std(self.mar_recent_window)) if len(self.mar_recent_window) >= 10 else 0.0
            is_steady_yawn = mar_std < 0.14

            if yawn_elapsed >= self.yawn_time and is_steady_yawn:
                is_yawning = True
                if not self._yawn_counted_in_current_bout:
                    self.yawn_count += 1
                    self._yawn_counted_in_current_bout = True
                if alert_level < 2:
                    status_text = "CHU Y: TAI XE DANG NGAP!"
                    status_color = (0, 215, 255)
                    alert_level = max(alert_level, 1)
        elif mar < self.mar_thresh * 0.85:
            # Hysteresis (ngưỡng trễ 85%) để tránh nhảy trạng thái khi miệng hơi khép nhẹ lúc đang ngáp
            self.yawn_start = None
            self._yawn_counted_in_current_bout = False

        # E. Kiểm tra mệt mỏi tích lũy qua PERCLOS & Micro-sleep
        if alert_level < 2:
            if perclos >= self.perclos_thresh or self.microsleep_count >= 3:
                status_text = "CANH BAO: MET MOI NGHIEM TRONG!"
                status_color = (0, 0, 255)
                alert_level = 2
            elif perclos >= self.perclos_warn_thresh or self.yawn_count >= 3:
                if alert_level == 0:
                    status_text = "NHAC NHO: DAU HIEU MET MOI!"
                    status_color = (0, 200, 255)
                    alert_level = 1

        # Tính thanh tiến trình nguy hiểm (Danger Progress Bar 0.0 -> 1.0)
        danger_progress = min(
            1.0,
            max(
                sleep_elapsed / max(0.1, self.sleep_time),
                nod_elapsed / max(0.1, self.nod_time),
                distract_elapsed / max(0.1, self.distraction_time),
                perclos / max(0.1, self.perclos_thresh),
            ),
        )

        return DriverStatus(
            is_calibrating=False,
            calib_time_left=0,
            status_text=status_text,
            status_color=status_color,
            alert_level=alert_level,
            trigger_alarm=(alert_level >= 2),
            danger_progress=danger_progress,
            is_yawning=is_yawning,
            yawn_count=self.yawn_count,
            blink_count=self.blink_count,
            microsleep_count=self.microsleep_count,
            ear=ear,
            mar=mar,
            eye_closed_prob=max(0.0, eye_closed_prob),
            pitch=delta_pitch,
            yaw=yaw - self.baseline_yaw,
            ear_thresh=self.ear_thresh,
            mar_thresh=self.mar_thresh,
            pitch_thresh=self.pitch_thresh,
            yaw_thresh=self.yaw_thresh,
            perclos=perclos,
            sleep_elapsed=sleep_elapsed,
            yawn_elapsed=yawn_elapsed,
            nod_elapsed=nod_elapsed,
            distract_elapsed=distract_elapsed,
        )
