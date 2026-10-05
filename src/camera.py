import cv2
from threading import Thread

class WebcamStream:
    def __init__(self, src):
        # src có thể là 0 (webcam laptop), 1 (camera rời), hoặc "video.mp4" (để test)
        self.stream = cv2.VideoCapture(src)
        
        # Ép độ phân giải ổn định để AI không bị quá tải
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

        if not self.grabbed:
            print(f"[LOI] Khong the ket noi voi camera: {src}")

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()