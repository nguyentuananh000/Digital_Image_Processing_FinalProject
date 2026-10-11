import time
from threading import Thread, Lock
import cv2


class WebcamStream:
    """
    Đọc luồng hình ảnh từ DroidCam Virtual Camera / Webcam / Video bằng luồng riêng (Multi-threading).
    Đã tối ưu riêng cho driver DroidCam trên Windows (MSMF):
      - Không mở/đóng lặp lại thiết bị để tránh lỗi khóa luồng MSMF (-1072875772).
      - Tự động bỏ qua khung hình chờ 'Start DroidCam' ban đầu để nhận luồng video thật từ điện thoại.
      - Hỗ trợ đổi nguồn camera nóng (phím V) và xoay khung hình 90/180/270 độ (phím R).
    """

    ROTATIONS = [0, 90, 180, 270]

    def __init__(self, src=1, resolution=(640, 480), rotation: int = 0, flip_horizontal: bool = False):
        self.src = src
        self.resolution = resolution
        self.rotation = rotation if rotation in self.ROTATIONS else 0
        self.flip_horizontal = flip_horizontal
        self._lock = Lock()
        self.stopped = False
        self.grabbed = False
        self.frame = None
        self.stream = None

        self._open_stream(self.src)

    def _open_stream(self, src):
        if self.stream is not None:
            try:
                self.stream.release()
            except Exception:
                pass
            time.sleep(0.3)

        self.src = src
        if isinstance(src, int):
            self.stream = cv2.VideoCapture(src, cv2.CAP_MSMF)
            if not self.stream.isOpened():
                self.stream = cv2.VideoCapture(src)
        else:
            self.stream = cv2.VideoCapture(src)

        if not self.stream.isOpened():
            print(f"[LOI] Khong the ket noi voi camera/nguon: {src}")
            with self._lock:
                self.grabbed = False
                self.frame = None
            return

        # Làm nóng luồng (Warm-up): Đọc bỏ qua vài khung hình đầu ('Start DroidCam' splash screen)
        grabbed, frame = False, None
        for _ in range(25):
            ret, f = self.stream.read()
            if ret and f is not None:
                grabbed, frame = True, f
                # Nếu khung hình đã có độ sáng thực tế (vượt qua màn hình chờ đen 'Start DroidCam')
                if f.mean() > 15.0:
                    break
            time.sleep(0.02)

        if grabbed and frame is not None:
            frame = self._transform_frame(frame)

        with self._lock:
            self.grabbed = grabbed
            self.frame = frame

        if grabbed:
            print(f"[Camera] Da ket noi thanh cong voi Camera {src} (DroidCam/Webcam)!")
        else:
            print(f"[CANH BAO] Khong doc duoc khung hinh tu Camera {src}.")

    def _transform_frame(self, frame):
        if self.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return frame

    def cycle_rotation(self) -> int:
        with self._lock:
            idx = (self.ROTATIONS.index(self.rotation) + 1) % len(self.ROTATIONS)
            self.rotation = self.ROTATIONS[idx]
            return self.rotation

    def switch_source(self, new_src) -> bool:
        """Đổi nóng nguồn camera (giữa DroidCam 1 và Webcam 0)."""
        print(f"[Camera] Dang chuyen sang Camera {new_src}...")
        self._open_stream(new_src)
        return self.grabbed

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.stream is None or not self.stream.isOpened():
                time.sleep(0.05)
                continue
            grabbed, frame = self.stream.read()
            if grabbed and frame is not None:
                frame = self._transform_frame(frame)
                with self._lock:
                    self.grabbed = True
                    self.frame = frame
                time.sleep(0.002)
            else:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.stopped = True
        time.sleep(0.05)
        if self.stream is not None and self.stream.isOpened():
            self.stream.release()
