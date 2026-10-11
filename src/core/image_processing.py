import math
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class DIPDiagnostics:
    """Thông tin chẩn đoán quá trình Tiền xử lý Ảnh số (Digital Image Processing) trên mỗi frame."""
    mode: str = "auto"               # 'auto', 'clahe', 'off'
    condition: str = "NORMAL"        # 'LOW-LIGHT', 'NORMAL', 'GLARE'
    mean_luminance: float = 128.0
    gamma: float = 1.0
    clahe_applied: bool = False


class AdaptiveImageProcessor:
    """
    Bộ tiền xử lý ảnh thích nghi ánh sáng (Adaptive Digital Image Processing Pipeline):
    - Phân tích độ sáng trung bình (Mean Luminance) trên kênh L của không gian màu CIE-LAB.
    - Tự động điều chỉnh Gamma (Adaptive Gamma Correction qua Look-Up Table tốc độ cao)
      kết hợp cân bằng histogram cục bộ CLAHE khi thiếu sáng (ban đêm) hoặc chói sáng (ngược nắng).
    """

    MODES = ["auto", "clahe", "off"]

    def __init__(self, mode: str = "auto", low_light_thresh: float = 85.0, glare_thresh: float = 195.0):
        self.mode = mode.lower() if mode.lower() in self.MODES else "auto"
        self.low_light_thresh = low_light_thresh
        self.glare_thresh = glare_thresh

        self._clahe_strong = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._clahe_normal = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._clahe_gentle = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

        # Cache bảng tra cứu LUT cho Gamma để tối ưu tốc độ thời gian thực
        self._lut_cache = {}

    def cycle_mode(self) -> str:
        idx = (self.MODES.index(self.mode) + 1) % len(self.MODES)
        self.mode = self.MODES[idx]
        return self.mode

    def _get_gamma_lut(self, gamma: float) -> np.ndarray:
        gamma_key = round(gamma, 2)
        if gamma_key not in self._lut_cache:
            inv_gamma = 1.0 / max(gamma_key, 0.1)
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            self._lut_cache[gamma_key] = table
        return self._lut_cache[gamma_key]

    def process(self, frame: np.ndarray, face_bbox=None) -> tuple[np.ndarray, DIPDiagnostics]:
        """
        Xử lý khung hình tốc độ cao (<0.15ms ở điều kiện sáng chuẩn) và trả về (ảnh đã xử lý, DIPDiagnostics).
        Nếu có `face_bbox` từ frame trước, đo độ sáng trực tiếp trên vùng khuôn mặt để chống hiện tượng Ngược sáng (Backlight).
        """
        h, w = frame.shape[:2]
        roi = frame
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if (x2 - x1) > 30 and (y2 - y1) > 30:
                roi = frame[y1:y2, x1:x2]

        # Đo nhanh độ sáng trung bình trên lưới lấy mẫu 1/8 của vùng quan tâm (ROI khuôn mặt)
        small_bgr = roi[::8, ::8]
        mean_l = float(
            0.114 * np.mean(small_bgr[:, :, 0])
            + 0.587 * np.mean(small_bgr[:, :, 1])
            + 0.299 * np.mean(small_bgr[:, :, 2])
        )

        if mean_l < self.low_light_thresh:
            condition = "LOW-LIGHT"
        elif mean_l > self.glare_thresh:
            condition = "GLARE"
        else:
            condition = "NORMAL"

        # Chế độ 1: Tắt tiền xử lý hoặc Ánh sáng chuẩn trong chế độ Auto -> Không tốn thời gian chuyển đổi LAB
        if self.mode == "off" or (self.mode == "auto" and condition == "NORMAL"):
            return frame, DIPDiagnostics(
                mode=self.mode,
                condition=condition,
                mean_luminance=mean_l,
                gamma=1.0,
                clahe_applied=False,
            )

        # Chỉ khi cần can thiệp ánh sáng (LOW-LIGHT, GLARE hoặc ép bật CLAHE) mới tách kênh LAB trên ảnh gốc
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Chế độ 2: Luôn bật CLAHE cố định
        if self.mode == "clahe":
            cl = self._clahe_normal.apply(l_channel)
            merged = cv2.merge((cl, a_channel, b_channel))
            out_frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            return out_frame, DIPDiagnostics(
                mode="clahe",
                condition=condition,
                mean_luminance=mean_l,
                gamma=1.0,
                clahe_applied=True,
            )

        # Chế độ 3: Tự động thích nghi khi Thiếu sáng (LOW-LIGHT) hoặc Chói sáng (GLARE)
        if condition == "LOW-LIGHT":
            gamma = 1.0 + min(0.85, (self.low_light_thresh - mean_l) / self.low_light_thresh)
            l_corrected = cv2.LUT(l_channel, self._get_gamma_lut(gamma))
            l_enhanced = self._clahe_strong.apply(l_corrected)
            l_enhanced = cv2.GaussianBlur(l_enhanced, (3, 3), 0)
            merged = cv2.merge((l_enhanced, a_channel, b_channel))
            out_frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            return out_frame, DIPDiagnostics(
                mode="auto",
                condition=condition,
                mean_luminance=mean_l,
                gamma=gamma,
                clahe_applied=True,
            )

        # condition == "GLARE"
        gamma = max(0.60, 1.0 - (mean_l - self.glare_thresh) / 150.0)
        l_corrected = cv2.LUT(l_channel, self._get_gamma_lut(gamma))
        l_enhanced = self._clahe_gentle.apply(l_corrected)
        merged = cv2.merge((l_enhanced, a_channel, b_channel))
        out_frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return out_frame, DIPDiagnostics(
            mode="auto",
            condition=condition,
            mean_luminance=mean_l,
            gamma=gamma,
            clahe_applied=True,
        )


class OneEuroFilter1D:
    """
    Bộ lọc tín hiệu thích nghi One-Euro Filter 1D cho chuỗi thời gian (EAR, MAR, Pitch, Yaw):
    - Khi tín hiệu ít biến đổi (tài xế giữ nguyên trạng thái): lọc mạnh để triệt tiêu nhiễu rung lắc xe.
    - Khi tín hiệu thay đổi đột ngột (nhắm mắt nhanh, gục đầu): tự động tăng tần số cắt để giảm độ trễ (lag) về ~0.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.15, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def filter(self, x: float, t: float) -> float:
        if self.t_prev is None or self.x_prev is None:
            self.x_prev = float(x)
            self.t_prev = float(t)
            return float(x)

        t_e = max(1e-4, float(t - self.t_prev))

        # Ước lượng đạo hàm (tốc độ biến thiên của tín hiệu)
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (float(x) - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Điều chỉnh tần số cắt thích nghi theo tốc độ biến thiên
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * float(x) + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = float(t)
        return x_hat
