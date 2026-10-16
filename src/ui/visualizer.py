from typing import Optional
import cv2
import numpy as np

from ..core.detector import DetectionResult
from ..core.image_processing import DIPDiagnostics
from ..core.tracker_logic import DriverStatus


class DMSVisualizer:
    """Vẽ giao diện HUD trực quan: điểm mốc mắt/miệng, thanh tiến trình nguy hiểm, thông số sinh trắc & Xử lý ảnh số."""

    @staticmethod
    def draw_hud(
        frame: np.ndarray,
        detection: DetectionResult,
        status: DriverStatus,
        fps: float,
        dip_info: Optional[DIPDiagnostics] = None,
        cam_info: str = "",
        show_text_overlay: bool = True,
    ) -> np.ndarray:
        h, w, _ = frame.shape

        # 1. Vẽ khung khuôn mặt & 16 điểm mốc (Mắt trái, Mắt phải, Miệng)
        cmp_res = detection.compare_result
        if cmp_res is not None and cmp_res.detected:
            # Vẽ điểm mốc MediaPipe (Màu Xanh lơ / Cam) để đối chiếu trực quan với Unified YOLO
            for pts_arr in (cmp_res.left_eye, cmp_res.right_eye, cmp_res.mouth):
                if pts_arr is not None:
                    pts_i = np.round(pts_arr).astype(np.int32)
                    cv2.polylines(frame, [pts_i], True, (255, 200, 0), 1, cv2.LINE_AA)
                    for x, y in pts_i:
                        if x > 0 and y > 0:
                            cv2.circle(frame, (int(x), int(y)), 2, (0, 165, 255), -1, cv2.LINE_AA)

        if detection.detected:
            if detection.bbox is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in detection.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)

            if detection.left_eye is not None:
                pts_le = np.round(detection.left_eye).astype(np.int32)
                cv2.polylines(frame, [pts_le], True, (0, 255, 255), 1, cv2.LINE_AA)
                for x, y in pts_le:
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1, cv2.LINE_AA)

            if detection.right_eye is not None:
                pts_re = np.round(detection.right_eye).astype(np.int32)
                cv2.polylines(frame, [pts_re], True, (0, 255, 255), 1, cv2.LINE_AA)
                for x, y in pts_re:
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1, cv2.LINE_AA)

            if detection.mouth is not None:
                pts_m = np.round(detection.mouth).astype(np.int32)
                cv2.polylines(frame, [pts_m], True, (255, 0, 255), 1, cv2.LINE_AA)
                for x, y in pts_m:
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 2, (255, 100, 255), -1, cv2.LINE_AA)


        # 4. Viền cảnh báo theo cấp độ (Level 1: Viền vàng/cam, Level 2: Viền đỏ dày)
        if status.alert_level >= 2:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        elif status.alert_level == 1:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 180, 255), 3)

        # Nếu tắt lớp phủ chữ (như trong Dashboard GUI), chỉ vẽ thêm bảng so sánh nếu đang ở mode COMPARE
        if not show_text_overlay:
            if cmp_res is not None:
                bx = w - 280
                cv2.rectangle(frame, (bx - 5, 10), (w - 10, 180), (20, 20, 20), -1)
                cv2.putText(frame, "SO SANH (COMPARE)", (bx, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2)
                y_ecnn = (detection.eye_closed_prob or 0.0) * 100.0
                m_ecnn = (cmp_res.eye_closed_prob or 0.0) * 100.0
                rows = [
                    f"EAR    : {detection.ear:.2f} vs {cmp_res.ear:.2f}",
                    f"EyeCNN : {y_ecnn:2.0f}% vs {m_ecnn:2.0f}%",
                    f"MAR    : {detection.mar:.2f} vs {cmp_res.mar:.2f}",
                    f"Speed  : {detection.latency_ms:.0f}ms vs {cmp_res.latency_ms:.0f}ms",
                    "Mau    : Xanh(YOLO) | Cam(MP)",
                ]
                for idx, r_txt in enumerate(rows):
                    cv2.putText(frame, r_txt, (bx, 58 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
            return frame

        # 2. Bảng thông tin Sinh trắc & Cảnh báo (Góc trái trên) cho chế độ CLI
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (335, 245), (20, 20, 20), -1)
        if dip_info is not None:
            cv2.rectangle(overlay, (w - 265, 10), (w - 10, 130), (20, 20, 20), -1)
        if cmp_res is not None:
            cv2.rectangle(overlay, (w - 285, 140), (w - 10, 315), (15, 25, 35), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Trạng thái chính
        cv2.putText(
            frame,
            status.status_text,
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status.status_color,
            2,
        )

        # Thanh tiến trình Nguy hiểm / Hiệu chuẩn (Progress Bar)
        bar_x, bar_y, bar_w, bar_h = 20, 48, 295, 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
        fill_w = int(bar_w * max(0.0, min(1.0, status.danger_progress)))
        if fill_w > 0:
            bar_color = (0, 255, 255) if status.is_calibrating else status.status_color
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        # Các chỉ số toán học
        ear_color = (0, 0, 255) if (status.ear > 0 and status.ear < status.ear_thresh) else (0, 255, 0)
        mar_color = (0, 215, 255) if status.mar > status.mar_thresh else (0, 255, 0)
        cnn_color = (0, 0, 255) if status.eye_closed_prob >= 0.58 else (0, 255, 0)

        cv2.putText(
            frame,
            f"EAR: {status.ear:.2f} (Thresh: {status.ear_thresh:.2f})",
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            ear_color,
            2,
        )
        cv2.putText(
            frame,
            f"Eye CNN (Sleepy): {status.eye_closed_prob * 100:.0f}%",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            cnn_color,
            2,
        )
        cv2.putText(
            frame,
            f"MAR: {status.mar:.2f} (Thresh: {status.mar_thresh:.2f})",
            (20, 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            mar_color,
            2,
        )
        cv2.putText(
            frame,
            f"Pitch: {status.pitch:+.1f} | Yaw: {status.yaw:+.1f}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (210, 210, 210),
            1,
        )
        cv2.putText(
            frame,
            f"PERCLOS: {status.perclos * 100:.1f}% | Micro-sleep: {status.microsleep_count}",
            (20, 168),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            frame,
            f"Chop mat: {status.blink_count} | Ngap: {status.yawn_count}",
            (20, 191),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 220),
            1,
        )
        model_label = f"{detection.model_source}" if cmp_res is not None else (
            f"{detection.model_source} + EyeCNN" if detection.detected else "Searching..."
        )
        cv2.putText(
            frame,
            f"AI: {model_label}",
            (20, 216),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (180, 220, 255),
            1,
        )

        # 3. Bảng thông tin Xử lý Ảnh số & Camera (Góc phải trên)
        if dip_info is not None:
            cond_color = (0, 255, 0) if dip_info.condition == "NORMAL" else (0, 215, 255)
            cv2.putText(
                frame,
                f"DIP Mode: {dip_info.mode.upper()}",
                (w - 255, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (180, 220, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Light: {dip_info.condition} (L={dip_info.mean_luminance:.0f})",
                (w - 255, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                cond_color,
                1,
            )
            cv2.putText(
                frame,
                f"Gamma: {dip_info.gamma:.2f} | CLAHE: {'ON' if dip_info.clahe_applied else 'OFF'}",
                (w - 255, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (210, 210, 210),
                1,
            )
            if cam_info:
                cv2.putText(
                    frame,
                    cam_info,
                    (w - 255, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (180, 255, 180),
                    1,
                )

        # 3B. Bảng Đối chiếu Trực tiếp (Khi ở chế độ COMPARE: Unified YOLO vs MediaPipe)
        if cmp_res is not None:
            bx = w - 275
            cv2.putText(
                frame,
                "SO SANH: UNIFIED vs MEDIAPIPE",
                (bx, 162),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 255, 255),
                2,
            )
            y_ecnn = (detection.eye_closed_prob or 0.0) * 100.0
            m_ecnn = (cmp_res.eye_closed_prob or 0.0) * 100.0
            rows = [
                f"EAR    : {detection.ear:.2f}   vs   {cmp_res.ear:.2f}",
                f"EyeCNN : {y_ecnn:3.0f}%   vs   {m_ecnn:3.0f}%",
                f"MAR    : {detection.mar:.2f}   vs   {cmp_res.mar:.2f}",
                f"Pitch  : {(detection.pitch or 0.0):+4.1f}  vs  {(cmp_res.pitch or 0.0):+4.1f}",
                f"Speed  : {detection.latency_ms:3.0f}ms  vs  {cmp_res.latency_ms:3.0f}ms",
                "Mau    : XanhLa(YOLO) | Cam(MP)",
            ]
            for idx, r_txt in enumerate(rows):
                cv2.putText(
                    frame,
                    r_txt,
                    (bx, 188 + idx * 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (230, 230, 230) if idx < 5 else (150, 220, 255),
                    1,
                )

        # 4. Viền cảnh báo theo cấp độ (Level 1: Viền vàng/cam, Level 2: Viền đỏ dày)
        if status.alert_level >= 2:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        elif status.alert_level == 1:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 180, 255), 3)

        # 5. Hiển thị FPS & phím tắt ở thanh dưới cùng
        cv2.putText(
            frame,
            f"FPS: {int(fps)} | [M]: Doi Mode (Unified/MediaPipe/SoSanh) | [V]: Cam | [R]: Xoay | [C]: Calib | [P]: DIP | [Q]: Thoat",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (220, 220, 220),
            1,
        )

        return frame
