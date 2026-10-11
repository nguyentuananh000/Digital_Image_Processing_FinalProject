import os
import time
from dataclasses import dataclass
from threading import Thread, Lock
from typing import Optional, Tuple
import cv2
import numpy as np

from .math_utils import calculate_ear, combine_bino_ear, calculate_mar, get_head_pose, estimate_pose_from_16kpts
from .eye_classifier import EyeStateClassifier


@dataclass
class DetectionResult:
    """Cấu trúc dữ liệu chuẩn hóa đầu ra của mọi bộ nhận diện (YOLOv8-Pose, Ensemble, MediaPipe)."""
    detected: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None
    left_eye: Optional[np.ndarray] = None      # shape (6, 2)
    right_eye: Optional[np.ndarray] = None     # shape (6, 2)
    mouth: Optional[np.ndarray] = None         # shape (4, 2)
    ear: float = 0.0
    mar: float = 0.0
    eye_closed_prob: Optional[float] = None    # Xac suat nham mat Tang 2 (CNN tren MRL Eye Dataset)
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    roll: Optional[float] = None
    conf: float = 0.0
    model_source: str = ""
    latency_ms: float = 0.0
    compare_result: Optional["DetectionResult"] = None


class FaceLandmarkDetector:
    """
    Bộ nhận diện khuôn mặt & điểm mốc thông minh (Kiến trúc Lai Đa Tầng):
    - 'mediapipe' (Mặc định): MediaPipe 3D FaceLandmarker + EyeStateNet CNN + Tự động cứu hộ bằng YOLO (YOLO-Rescue) khi gục sâu/che mặt.
    - 'unified': Model YOLOv8-Pose hợp nhất (models/dms_unified.pt) + EyeStateNet CNN.
    - 'compare': Chạy song song cả MediaPipe và Unified YOLO trên cùng khung hình để đối chiếu trực tiếp.
    """

    LEFT_EYE_MP_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_MP_IDX = [362, 385, 387, 263, 373, 380]
    MOUTH_MP_IDX = [78, 13, 308, 14]
    HEAD_POSE_MP_IDX = [1, 152, 33, 263, 61, 291]

    def __init__(
        self,
        backend: str = "mediapipe",
        unified_weights: str = "models/dms_unified.pt",
        eye_classifier_weights: str = "models/eye_classifier.pt",
        conf_threshold: float = 0.5,
        kpt_conf_threshold: float = 0.3,
        imgsz: int = 416,
    ):
        self.backend = backend.lower()
        self.unified_weights = unified_weights
        self.conf_threshold = conf_threshold
        self.kpt_conf_threshold = kpt_conf_threshold
        self.imgsz = int(imgsz)
        self.primary_model = None
        self.primary_name = ""
        self.face_mesh = None
        self._mp_use_tasks = False
        self.eye_classifier = EyeStateClassifier(
            weights_path=eye_classifier_weights,
            unified_path=unified_weights,
        )
        self.set_backend(self.backend)

    def _ensure_yolo_loaded(self):
        if self.primary_model is not None:
            return
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            torch.set_num_threads(4)

        if not os.path.exists(self.unified_weights):
            raise FileNotFoundError(f"Khong tim thay file trong so YOLO tai: {self.unified_weights}")

        print(f"[Detector] Dang tai model YOLO: {self.unified_weights} (imgsz={self.imgsz})")
        self.primary_model = YOLO(self.unified_weights)
        self.primary_name = os.path.basename(self.unified_weights)
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.primary_model(dummy, imgsz=self.imgsz, verbose=False)

    def _ensure_mediapipe_loaded(self):
        if self.face_mesh is not None:
            return
        import mediapipe as mp

        self._mp_use_tasks = False
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=self.conf_threshold,
                min_tracking_confidence=self.conf_threshold,
            )
        elif hasattr(mp, "tasks"):
            import urllib.request
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._mp_use_tasks = True
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            task_path = os.path.join(root_dir, "models", "face_landmarker.task")
            if not os.path.exists(task_path):
                os.makedirs(os.path.dirname(task_path), exist_ok=True)
                url = (
                    "https://storage.googleapis.com/mediapipe-models/"
                    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                )
                print(f"[Detector] Dang tai MediaPipe model: {task_path}...")
                urllib.request.urlretrieve(url, task_path)

            print(f"[Detector] Dang khoi tao MediaPipe FaceLandmarker: {task_path}")
            base_options = mp_python.BaseOptions(model_asset_path=task_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                output_face_blendshapes=True,
                min_face_detection_confidence=self.conf_threshold,
                min_face_presence_confidence=self.conf_threshold,
                min_tracking_confidence=self.conf_threshold,
            )
            self.face_mesh = vision.FaceLandmarker.create_from_options(options)
        else:
            raise RuntimeError("Khong the khoi tao MediaPipe FaceMesh / FaceLandmarker.")

    def set_backend(self, backend: str):
        """Chuyển đổi chế độ tức thì ('mediapipe', 'unified', 'compare') mà không cần nạp lại model từ ổ cứng."""
        b = backend.lower()
        if b == "yolo":
            b = "unified"
        if b == "unified":
            self._ensure_yolo_loaded()
        elif b == "mediapipe":
            self._ensure_mediapipe_loaded()
        elif b == "compare":
            self._ensure_yolo_loaded()
            self._ensure_mediapipe_loaded()
        else:
            raise ValueError(f"Backend khong hop le: {backend}. Chon 'mediapipe', 'unified', hoac 'compare'.")
        self.backend = b

    def _extract_yolo_single(self, model, frame: np.ndarray):
        """Trích xuất bbox (x1,y1,x2,y2), conf, tọa độ 16 điểm (16,2) và độ tự tin từng điểm (16,) từ model YOLO."""
        results = model(frame, imgsz=self.imgsz, conf=self.conf_threshold, verbose=False)
        if not results or len(results) == 0:
            return None

        res = results[0]
        if res.keypoints is None or res.keypoints.xy is None:
            return None

        if len(res.keypoints.xy) == 0 or len(res.keypoints.xy[0]) != 16:
            return None

        pts_xy = res.keypoints.xy[0].cpu().numpy()  # (16, 2)
        if res.keypoints.conf is not None and len(res.keypoints.conf) > 0:
            pts_conf = res.keypoints.conf[0].cpu().numpy()  # (16,)
        else:
            pts_conf = np.ones(16, dtype=np.float32)

        bbox = None
        box_conf = float(np.mean(pts_conf))
        if res.boxes is not None and len(res.boxes.xyxy) > 0:
            xyxy = res.boxes.xyxy[0].cpu().numpy().astype(int)
            bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
            if res.boxes.conf is not None and len(res.boxes.conf) > 0:
                box_conf = float(res.boxes.conf[0].cpu().item())

        return {
            "bbox": bbox,
            "box_conf": box_conf,
            "pts_xy": pts_xy,
            "pts_conf": pts_conf,
        }

    def _process_mediapipe(self, frame: np.ndarray) -> DetectionResult:
        import mediapipe as mp

        t0 = time.perf_counter()
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_blink_prob = None

        if getattr(self, "_mp_use_tasks", False):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.face_mesh.detect(mp_image)
            if not results.face_landmarks:
                return DetectionResult(detected=False, model_source="MediaPipe", latency_ms=(time.perf_counter() - t0) * 1000.0)
            landmarks_list = results.face_landmarks[0]
            if getattr(results, "face_blendshapes", None) and len(results.face_blendshapes) > 0:
                bs_map = {b.category_name: float(b.score) for b in results.face_blendshapes[0]}
                if "eyeBlinkLeft" in bs_map and "eyeBlinkRight" in bs_map:
                    mp_blink_prob = 0.5 * (bs_map["eyeBlinkLeft"] + bs_map["eyeBlinkRight"])
        else:
            results = self.face_mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                return DetectionResult(detected=False, model_source="MediaPipe", latency_ms=(time.perf_counter() - t0) * 1000.0)
            landmarks_list = results.multi_face_landmarks[0].landmark

        lm = [(int(pt.x * w), int(pt.y * h)) for pt in landmarks_list]

        left_eye = np.array([lm[i] for i in self.LEFT_EYE_MP_IDX], dtype=np.float32)
        right_eye = np.array([lm[i] for i in self.RIGHT_EYE_MP_IDX], dtype=np.float32)
        mouth = np.array([lm[i] for i in self.MOUTH_MP_IDX], dtype=np.float32)

        image_pts = np.array([lm[i] for i in self.HEAD_POSE_MP_IDX], dtype="double")
        pitch, yaw, roll = get_head_pose(frame.shape, image_pts)

        ear = combine_bino_ear(calculate_ear(left_eye), calculate_ear(right_eye), yaw=yaw or 0.0)
        mar = calculate_mar(mouth)
        cnn_prob = self.eye_classifier.predict_from_landmarks(
            frame, left_eye, right_eye, yaw=yaw or 0.0
        )
        if cnn_prob is not None and mp_blink_prob is not None:
            eye_closed_prob = 0.55 * cnn_prob + 0.45 * mp_blink_prob
        else:
            eye_closed_prob = cnn_prob if cnn_prob is not None else mp_blink_prob

        xs = [p[0] for p in lm]
        ys = [p[1] for p in lm]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return DetectionResult(
            detected=True,
            bbox=bbox,
            left_eye=left_eye,
            right_eye=right_eye,
            mouth=mouth,
            ear=ear,
            mar=mar,
            eye_closed_prob=eye_closed_prob,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            conf=1.0,
            model_source="MediaPipe",
            latency_ms=dt_ms,
        )

    def _process_yolo_primary(self, frame: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        out = self._extract_yolo_single(self.primary_model, frame)
        if out is None:
            return DetectionResult(
                detected=False,
                model_source=self.primary_name,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        res = self._build_result_from_kpts(
            frame, out["pts_xy"], out["pts_conf"], out["bbox"], out["box_conf"], self.primary_name
        )
        res.latency_ms = (time.perf_counter() - t0) * 1000.0
        return res

    def process(self, frame: np.ndarray) -> DetectionResult:
        """Phân tích khung hình và trả về DetectionResult chuẩn hóa."""
        if self.backend == "unified":
            return self._process_yolo_primary(frame)

        elif self.backend == "mediapipe":
            res_mp = self._process_mediapipe(frame)
            if not res_mp.detected:
                # Bọc lót tự động (Smart Fallback): Khi tài xế gục sâu hoặc quay đầu mạnh khiến MediaPipe mất dấu,
                # tự động gọi model YOLOv8-Pose đã train (dms_unified.pt) vào cứu hộ ngay lập tức!
                try:
                    self._ensure_yolo_loaded()
                    res_fb = self._process_yolo_primary(frame)
                    if res_fb.detected:
                        res_fb.model_source = f"YOLO-Rescue({self.primary_name})"
                        return res_fb
                except Exception:
                    pass
            return res_mp

        elif self.backend == "compare":
            # Chạy cả 2 model (Unified YOLO + MediaPipe) trên cùng khung hình để so sánh trực tiếp
            res_yolo = self._process_yolo_primary(frame)
            res_mp = self._process_mediapipe(frame)
            res_yolo.compare_result = res_mp
            res_yolo.model_source = "COMPARE (Unified vs MediaPipe)"
            return res_yolo

        return DetectionResult(detected=False)

    def _build_result_from_kpts(
        self,
        frame: np.ndarray,
        pts_xy: np.ndarray,
        pts_conf: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
        box_conf: float,
        source_name: str,
    ) -> DetectionResult:
        """Tạo DetectionResult từ 16 điểm mốc YOLOv8-Pose + Tầng 2 Eye CNN."""
        left_eye = pts_xy[0:6]
        right_eye = pts_xy[6:12]
        mouth = pts_xy[12:16]

        left_conf = float(np.mean(pts_conf[0:6]))
        right_conf = float(np.mean(pts_conf[6:12]))

        ear_left = calculate_ear(left_eye) if left_conf >= self.kpt_conf_threshold else 0.0
        ear_right = calculate_ear(right_eye) if right_conf >= self.kpt_conf_threshold else 0.0

        pitch, yaw, roll = estimate_pose_from_16kpts(frame.shape, pts_xy, bbox)
        avg_ear = combine_bino_ear(ear_left, ear_right, yaw=yaw or 0.0)
        mar = calculate_mar(mouth) if float(np.mean(pts_conf[12:16])) >= self.kpt_conf_threshold else 0.0

        eye_closed_prob = self.eye_classifier.predict_from_landmarks(
            frame,
            left_eye if left_conf >= self.kpt_conf_threshold else None,
            right_eye if right_conf >= self.kpt_conf_threshold else None,
            yaw=yaw or 0.0,
        )

        return DetectionResult(
            detected=True,
            bbox=bbox,
            left_eye=left_eye,
            right_eye=right_eye,
            mouth=mouth,
            ear=avg_ear,
            mar=mar,
            eye_closed_prob=eye_closed_prob,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            conf=box_conf,
            model_source=source_name,
        )


class AsyncDetectorWorker:
    """
    Luồng suy luận AI bất đồng bộ (Asynchronous AI Inference Thread):
    Tách biệt hoàn toàn thời gian chạy mạng nơ-ron YOLOv8 khỏi vòng lặp hiển thị Video,
    giúp giao diện và luồng Camera luôn đạt chuẩn mượt 30 FPS ngay cả khi chạy trên CPU.
    """

    def __init__(self, detector: FaceLandmarkDetector):
        self.detector = detector
        self._lock = Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._has_new_frame = False
        self._latest_result = DetectionResult(detected=False)
        self._stopped = False
        self.ai_fps = 0.0

    def start(self):
        Thread(target=self._run, args=(), daemon=True).start()
        return self

    def set_detector(self, new_detector: FaceLandmarkDetector):
        with self._lock:
            self.detector = new_detector

    def submit_frame(self, frame: np.ndarray):
        with self._lock:
            self._latest_frame = frame
            self._has_new_frame = True

    def get_result(self) -> DetectionResult:
        with self._lock:
            return self._latest_result

    def _run(self):
        while not self._stopped:
            frame_to_process = None
            detector_ref = None
            with self._lock:
                if self._has_new_frame and self._latest_frame is not None:
                    frame_to_process = self._latest_frame
                    self._has_new_frame = False
                    detector_ref = self.detector

            if frame_to_process is None or detector_ref is None:
                time.sleep(0.003)
                continue

            t0 = time.perf_counter()
            try:
                res = detector_ref.process(frame_to_process)
                dt = time.perf_counter() - t0
                with self._lock:
                    self._latest_result = res
                    self.ai_fps = (1.0 / dt) if dt > 0 else 0.0
            except Exception as e:
                print(f"[LOI AI Worker] {e}")
                time.sleep(0.05)

    def stop(self):
        self._stopped = True
