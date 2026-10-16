import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

import importlib.util

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Tự động chuyển sang môi trường Python đã cài đầy đủ thư viện (Python 3.13 / .venv)
# Dùng find_spec thay vì import trực tiếp để tiết kiệm 1.5s khởi động
_has_deps = (
    importlib.util.find_spec("cv2") is not None
    and importlib.util.find_spec("customtkinter") is not None
)
if not _has_deps:
    _CANDIDATE_PYTHONS = [
        os.path.expanduser(r"~\AppData\Local\Programs\Python\Python313\python.exe"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts", "python.exe")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")),
    ]
    for _py_path in _CANDIDATE_PYTHONS:
        if os.path.exists(_py_path) and os.path.normcase(_py_path) != os.path.normcase(sys.executable):
            print(f"[Env] Chuyen tu {sys.executable} sang moi truong da cai thu vien: {_py_path}")
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            sys.exit(subprocess.call([_py_path, *sys.argv], env=env))
    raise ModuleNotFoundError("Chua cai dat day du thu vien (cv2, customtkinter). Vui long cai dat truoc khi chay.")



AVAILABLE_MODES = ["mediapipe", "unified", "compare"]


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[LOI] Khong tim thay file cau hinh tai: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_detector(mode: str, model_cfg: dict, root_dir: str):
    from src.core.detector import FaceLandmarkDetector

    unified_weights = os.path.join(root_dir, model_cfg.get("unified_weights", "models/dms_unified.pt"))
    eye_weights = os.path.join(root_dir, model_cfg.get("eye_classifier_weights", "models/eye_classifier.pt"))
    conf_thresh = float(model_cfg.get("conf_threshold", 0.5))
    kpt_conf_thresh = float(model_cfg.get("kpt_conf_threshold", 0.3))
    imgsz = int(model_cfg.get("imgsz", 416))

    return FaceLandmarkDetector(
        backend=mode,
        unified_weights=unified_weights,
        eye_classifier_weights=eye_weights,
        conf_threshold=conf_thresh,
        kpt_conf_threshold=kpt_conf_thresh,
        imgsz=imgsz,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Hệ thống Giám sát Tài xế Buồn ngủ (DMS)")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=AVAILABLE_MODES + ["yolo"],
        help="Chọn chế độ AI: 'mediapipe' (Mặc định: MediaPipe + EyeCNN + YOLO-Rescue), 'unified' (dms_unified.pt), hoặc 'compare'",
    )
    parser.add_argument(
        "--dip",
        type=str,
        default=None,
        choices=["auto", "clahe", "off"],
        help="Chế độ tiền xử lý ảnh số: 'auto' (Gamma + CLAHE thích nghi ánh sáng), 'clahe' (luôn bật CLAHE), 'off' (tắt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Nguồn camera (ví dụ: 0, 1) hoặc URL/đường dẫn file video",
    )
    parser.add_argument(
        "--droidcam",
        type=str,
        default=None,
        help="Địa chỉ IP của DroidCam trên điện thoại (ví dụ: 192.168.1.82 hoặc 192.168.1.82:4747)",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=None,
        choices=[0, 90, 180, 270],
        help="Xoay khung hình camera điện thoại (0, 90, 180, 270 độ)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Tốc độ khung hình mục tiêu (mặc định: 30 FPS)",
    )
    parser.add_argument(
        "--no-calib",
        action="store_true",
        help="Bỏ qua bước hiệu chuẩn 5 giây đầu, dùng luôn ngưỡng mặc định trong config.json",
    )
    parser.add_argument(
        "--cli",
        "--no-gui",
        dest="cli",
        action="store_true",
        help="Chạy ở chế độ giao diện OpenCV truyền thống (mặc định khởi chạy Dashboard CustomTkinter trực quan)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root_dir, "config.json")

    # Mặc định khởi chạy Giao diện đồ họa trực quan (CustomTkinter Dashboard)
    # trừ khi người dùng chạy với tham số --cli hoặc --no-gui
    if not args.cli:
        from src.ui.app_gui import launch_gui
        launch_gui(config_path)
        return

    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(e)
        return

    # 1. Trích xuất cấu hình phần cứng & tiền xử lý
    hw_cfg = config.get("hardware", {})
    model_cfg = config.get("model", {})
    prep_cfg = config.get("preprocessing", {})
    thresh_cfg = config.get("thresholds", {}).copy()

    if args.no_calib:
        thresh_cfg["calibration_time"] = 0.0

    # Xác định nguồn camera (Mặc định trong config.json là 1 -> DroidCam Video)
    cam_src = hw_cfg.get("camera_source", 1)
    if args.droidcam:
        ip_str = args.droidcam.strip()
        if not ip_str.startswith("http"):
            if ":" not in ip_str:
                ip_str = f"{ip_str}:4747"
            cam_src = f"http://{ip_str}/video"
        else:
            cam_src = ip_str
    elif args.source is not None:
        cam_src = int(args.source) if args.source.isdigit() else args.source

    cam_res = tuple(hw_cfg.get("camera_resolution", [640, 480]))
    target_fps = args.fps if args.fps is not None else int(hw_cfg.get("target_fps", 30))
    frame_interval = 1.0 / max(1, target_fps)
    cam_rot = args.rotate if args.rotate is not None else int(hw_cfg.get("camera_rotation", 0))
    flip_h = bool(hw_cfg.get("flip_horizontal", False))

    alert_method = hw_cfg.get("alert_method", "speaker")
    audio_file = os.path.join(root_dir, hw_cfg.get("audio_file", "assets/alarm.wav"))

    dip_mode = args.dip or prep_cfg.get("dip_mode", "auto")
    low_light_thresh = float(prep_cfg.get("low_light_threshold", 85.0))
    glare_thresh = float(prep_cfg.get("glare_threshold", 195.0))
    smoothing_window = int(prep_cfg.get("smoothing_window", 5))

    # Xác định chế độ chạy AI (Mặc định: 'mediapipe' -> MediaPipe 3D + EyeCNN + YOLO-Rescue)
    current_mode = args.mode or model_cfg.get("backend", "mediapipe")
    if current_mode == "yolo":
        current_mode = "unified"

    # 2. Khởi tạo các mô-đun cốt lõi (3 luồng độc lập: Camera + AI Worker + UI 30 FPS)
    from src.core.camera import WebcamStream
    from src.core.alert_system import AlertSystem
    from src.core.image_processing import AdaptiveImageProcessor
    from src.core.detector import AsyncDetectorWorker
    from src.core.tracker_logic import DrowsinessTracker
    from src.ui.visualizer import DMSVisualizer

    alert = AlertSystem(method=alert_method, sound_path=audio_file)
    img_processor = AdaptiveImageProcessor(
        mode=dip_mode,
        low_light_thresh=low_light_thresh,
        glare_thresh=glare_thresh,
    )
    detector = build_detector(current_mode, model_cfg, root_dir)
    ai_worker = AsyncDetectorWorker(detector).start()
    tracker = DrowsinessTracker(thresholds_cfg=thresh_cfg, smoothing_window=smoothing_window)

    print(f"[Camera] Dang khoi dong tu nguon: {cam_src} (Muc tieu: {target_fps} FPS)...")
    vs = WebcamStream(src=cam_src, resolution=cam_res, rotation=cam_rot, flip_horizontal=flip_h).start()
    time.sleep(0.3)

    fps_history = deque(maxlen=20)
    prev_time = time.perf_counter()
    last_face_bbox = None
    print(
        "He thong DMS dang chay o 30 FPS!\n"
        "Phim tat: [V] Doi Camera (0/1) | [R] Xoay anh 90 do | [C] Hieu chuan | [M] Doi Model | [P] Doi DIP | [Q/ESC] Thoat."
    )

    # 3. Vòng lặp hiển thị & điều khiển khóa chuẩn 30 FPS
    try:
        while True:
            loop_start = time.perf_counter()

            success, frame = vs.read()
            if not success or frame is None:
                break

            # Tiền xử lý Ảnh số thích nghi ánh sáng theo đúng vùng mặt (Face-ROI Backlight Metering)
            proc_frame, dip_info = img_processor.process(frame, face_bbox=last_face_bbox)

            # Gửi khung hình mới nhất cho luồng AI xử lý ngầm (không gây khựng khung hình)
            ai_worker.submit_frame(proc_frame)
            detection = ai_worker.get_result()
            if detection.detected and detection.bbox is not None:
                last_face_bbox = detection.bbox

            # Cập nhật trạng thái buồn ngủ / ngáp / gục đầu / mất tập trung
            status = tracker.update(detection)

            # Kích hoạt cảnh báo âm thanh đa cấp độ (0: Safe, 1: Warning Beep, 2: Emergency Alarm)
            alert.update_level(status.alert_level)

            # Vẽ giao diện HUD
            fps = sum(fps_history) / len(fps_history) if fps_history else float(target_fps)
            display_frame = DMSVisualizer.draw_hud(
                proc_frame,
                detection,
                status,
                fps,
                dip_info,
                cam_info=f"Cam: {vs.src} | AI: {ai_worker.ai_fps:.0f}fps",
            )
            cv2.imshow("He Thong Giam Sat Tai Xe (DMS)", display_frame)

            # Xử lý phím tắt (1ms)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            elif key in (ord("c"), ord("C")):
                alert.stop()
                tracker.reset_calibration()
            elif key in (ord("p"), ord("P")):
                new_dip = img_processor.cycle_mode()
                print(f"[DIP] Da chuyen che do Tien xu ly Anh so sang: {new_dip.upper()}")
            elif key in (ord("r"), ord("R")):
                new_rot = vs.cycle_rotation()
                print(f"[Camera] Da xoay khung hinh: {new_rot} do")
            elif key in (ord("v"), ord("V")):
                if isinstance(vs.src, int):
                    next_cam = 1 if vs.src == 0 else 0
                    vs.switch_source(next_cam)
                    tracker.reset_calibration()
            elif key in (ord("m"), ord("M")):
                alert.stop()
                cur_idx = AVAILABLE_MODES.index(current_mode) if current_mode in AVAILABLE_MODES else 0
                next_mode = AVAILABLE_MODES[(cur_idx + 1) % len(AVAILABLE_MODES)]
                try:
                    detector.set_backend(next_mode)
                    current_mode = next_mode
                    print(f"[Model] Da chuyen sang che do: {current_mode.upper()}")
                except Exception as e:
                    print(f"[CANH BAO] Khong the chuyen sang mode '{next_mode}': {e}")

            # Điều nhịp khung hình (High-Precision Frame Pacer) chuẩn 30 FPS (33.33ms / frame)
            elapsed = time.perf_counter() - loop_start
            remaining = frame_interval - elapsed
            if remaining > 0.002:
                time.sleep(remaining - 0.002)
            while (time.perf_counter() - loop_start) < frame_interval:
                pass

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps_history.append(1.0 / dt)

    finally:
        alert.stop()
        ai_worker.stop()
        vs.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
