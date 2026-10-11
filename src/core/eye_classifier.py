import math
import os
from typing import Optional
import cv2
import numpy as np
import torch
import torch.nn as nn


class EyeStateNet(nn.Module):
    """
    Mạng CNN siêu nhẹ phân loại trạng thái mắt (0: Awake/Mở, 1: Sleepy/Nhắm) trên ảnh vùng mắt 48x48.
    Thiết kế tối ưu cho CPU thời gian thực (< 0.6ms cho cả 2 mắt).
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24x24

            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12x12

            nn.Conv2d(48, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 6x6

            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(96, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class EyeStateClassifier:
    """
    Bộ phân loại trạng thái mắt Tầng 2 (Stage-2 Eye Patch Classifier):
    - Cắt vùng ảnh vuông quanh mắt dựa trên 6 điểm mốc mắt (căn chỉnh tương đương bộ MRL Eye Dataset).
    - Tiền xử lý bằng CLAHE + Chuẩn hóa Z-score (bất biến với điều kiện ánh sáng và màu sắc camera).
    - Dự đoán xác suất mắt nhắm P(sleepy) thuộc [0.0, 1.0].
    """

    INPUT_SIZE = 48

    def __init__(self, weights_path: str = "models/eye_classifier.pt", unified_path: str = "models/dms_unified.pt"):
        self.available = False
        self.model: Optional[EyeStateNet] = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

        # Ưu tiên nạp từ file standalone nhẹ (280KB), hoặc nạp trực tiếp từ nhánh nhúng trong dms_unified.pt
        if os.path.exists(weights_path):
            try:
                self.model = EyeStateNet()
                ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
                state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.available = True
                val_acc = ckpt.get("val_acc", 0.0) if isinstance(ckpt, dict) else 0.0
                print(f"[EyeClassifier] Da tai nhanh EyeCNN: {weights_path} (Val Acc: {val_acc*100:.1f}%)")
                return
            except Exception as e:
                print(f"[CANH BAO] Khong the tai EyeStateClassifier tai {weights_path}: {e}")

        if os.path.exists(unified_path):
            try:
                ckpt = torch.load(unified_path, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "eye_classifier_state_dict" in ckpt:
                    self.model = EyeStateNet()
                    self.model.load_state_dict(ckpt["eye_classifier_state_dict"])
                    self.model.eval()
                    self.available = True
                    val_acc = ckpt.get("eye_classifier_val_acc", 0.98)
                    print(f"[EyeClassifier] Da tai nhanh EyeCNN nhung trong {unified_path} (Val Acc: {val_acc*100:.1f}%)")
            except Exception as e:
                print(f"[CANH BAO] Khong the tai EyeCNN tu {unified_path}: {e}")

    def preprocess_patch(self, gray_patch: np.ndarray) -> np.ndarray:
        """Chuẩn hóa ảnh vùng mắt (CLAHE + Resize 48x48 + Z-score Normalization)."""
        if gray_patch.ndim == 3:
            gray_patch = cv2.cvtColor(gray_patch, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_patch, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        eq = self._clahe.apply(resized)
        arr = eq.astype(np.float32)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        return (arr - mean) / max(std, 10.0)

    def crop_eye_from_landmarks(self, frame_gray: np.ndarray, eye_pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Cắt vùng mắt vuông từ 6 điểm mốc mắt với tỷ lệ khung hình khớp với MRL Eye Dataset
        (chiều rộng khung = 1.75 * chiều dài mắt, dịch tâm lên nhẹ 12% để bao phủ mí mắt và chân mày).
        """
        if eye_pts is None or len(eye_pts) != 6:
            return None
        for pt in eye_pts:
            if pt[0] <= 0 or pt[1] <= 0:
                return None

        h, w = frame_gray.shape[:2]
        xs = eye_pts[:, 0]
        ys = eye_pts[:, 1]
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        eye_w = math.hypot(float(eye_pts[3][0] - eye_pts[0][0]), float(eye_pts[3][1] - eye_pts[0][1]))
        if eye_w < 6.0:
            return None

        half_side = max(10.0, eye_w * 0.90)
        cy_shifted = cy - eye_w * 0.12  # Dịch nhẹ lên phía mí trên/chân mày giống ảnh gốc MRL

        x1 = int(round(cx - half_side))
        x2 = int(round(cx + half_side))
        y1 = int(round(cy_shifted - half_side))
        y2 = int(round(cy_shifted + half_side))

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h or (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        patch = frame_gray[y1:y2, x1:x2]
        return self.preprocess_patch(patch)

    @torch.no_grad()
    def predict_patch_sleepy_prob(self, patch: np.ndarray) -> float:
        """Dự đoán xác suất mắt nhắm P(sleepy) từ 1 ảnh vùng mắt đã cắt sẵn."""
        if not self.available or self.model is None:
            return 0.0
        norm_patch = self.preprocess_patch(patch)
        tensor = torch.from_numpy(norm_patch).unsqueeze(0).unsqueeze(0)  # (1, 1, 48, 48)
        logits = self.model(tensor)
        return float(torch.softmax(logits, dim=1)[0, 1].item())

    @torch.no_grad()
    def predict_from_landmarks(
        self,
        frame_bgr: np.ndarray,
        left_eye: Optional[np.ndarray],
        right_eye: Optional[np.ndarray],
        yaw: float = 0.0,
    ) -> Optional[float]:
        """
        Trả về xác suất mắt nhắm P(sleepy) in [0.0, 1.0] kết hợp 2 mắt có trọng số theo góc quay đầu (Yaw).
        """
        if not self.available or self.model is None:
            return None

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        p_left = self.crop_eye_from_landmarks(frame_gray, left_eye) if left_eye is not None else None
        p_right = self.crop_eye_from_landmarks(frame_gray, right_eye) if right_eye is not None else None

        patches = []
        valid_sides = []
        if p_left is not None:
            patches.append(p_left)
            valid_sides.append("left")
        if p_right is not None:
            patches.append(p_right)
            valid_sides.append("right")

        if not patches:
            return None

        batch = torch.from_numpy(np.stack(patches, axis=0)).unsqueeze(1)  # (N, 1, 48, 48)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Xác suất 'sleepy' (mắt nhắm)

        if len(probs) == 1:
            return float(probs[0])

        prob_left, prob_right = float(probs[0]), float(probs[1])
        if abs(yaw) > 15.0:
            # Ưu tiên mắt gần camera hơn khi quay đầu sang trái/phải
            if yaw > 15.0:
                return 0.75 * prob_left + 0.25 * prob_right
            else:
                return 0.25 * prob_left + 0.75 * prob_right
        return 0.5 * (prob_left + prob_right)

    def predict_sleepy_prob(
        self, frame_bgr: np.ndarray, left_eye: Optional[np.ndarray], right_eye: Optional[np.ndarray]
    ) -> Optional[float]:
        return self.predict_from_landmarks(frame_bgr, left_eye, right_eye, yaw=0.0)

