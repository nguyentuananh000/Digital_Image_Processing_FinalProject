import argparse
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.core.eye_classifier import EyeStateClassifier
from src.core.math_utils import calculate_ear, calculate_mar


def parse_points_str(pts_str: str) -> np.ndarray:
    """Chuyển chuỗi 'x1,y1;x2,y2;...' từ CVAT XML sang mảng numpy (N, 2)."""
    pts = []
    for token in pts_str.strip().split(";"):
        if not token:
            continue
        x_str, y_str = token.split(",")
        pts.append([float(x_str), float(y_str)])
    return np.array(pts, dtype=np.float32)


def pinch_eyelid_points(eye_pts: np.ndarray, pinch_ratio: float = 0.65) -> np.ndarray:
    """
    Sửa lỗi nhãn giả (Pseudo-label gap) của MediaPipe khi mắt nhắm:
    Kéo 2 cặp điểm mí trên (p2, p3) và mí dưới (p6, p5) sát lại đường trung tuyến mí mắt,
    giúp model YOLOv8-Pose học được hình thái mắt nhắm thật sự (EAR ~ 0.10 - 0.12).
    """
    pts = eye_pts.copy()
    # Cặp dọc 1: p2 (idx 1) và p6 (idx 5)
    mid_15 = 0.5 * (pts[1] + pts[5])
    pts[1] = mid_15 + (pts[1] - mid_15) * (1.0 - pinch_ratio)
    pts[5] = mid_15 + (pts[5] - mid_15) * (1.0 - pinch_ratio)

    # Cặp dọc 2: p3 (idx 2) và p5 (idx 4)
    mid_24 = 0.5 * (pts[2] + pts[4])
    pts[2] = mid_24 + (pts[2] - mid_24) * (1.0 - pinch_ratio)
    pts[4] = mid_24 + (pts[4] - mid_24) * (1.0 - pinch_ratio)
    return pts


def extract_xml_entries(xml_path: str, image_root: str, dataset_tag: str) -> List[dict]:
    """Đọc toàn bộ bản ghi hợp lệ (đủ left_eye, right_eye, mouth) từ file XML."""
    if not os.path.exists(xml_path):
        print(f"[CANH BAO] Khong tim thay file XML: {xml_path}")
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()
    entries = []

    for img_el in root.findall("image"):
        rel_name = img_el.attrib.get("name", "")
        w = int(float(img_el.attrib.get("width", 0)))
        h = int(float(img_el.attrib.get("height", 0)))
        if w <= 0 or h <= 0:
            continue

        pts_map = {}
        for pt_el in img_el.findall("points"):
            label = pt_el.attrib.get("label", "")
            pts_str = pt_el.attrib.get("points", "")
            if label in ("left_eye", "right_eye", "mouth") and pts_str:
                pts_map[label] = parse_points_str(pts_str)

        if len(pts_map) != 3:
            continue
        if len(pts_map["left_eye"]) != 6 or len(pts_map["right_eye"]) != 6 or len(pts_map["mouth"]) != 4:
            continue

        full_path = os.path.join(image_root, rel_name)
        is_drowsy_folder = "/Drowsy/" in rel_name.replace("\\", "/")
        mar_val = calculate_mar(pts_map["mouth"])
        ear_val = 0.5 * (calculate_ear(pts_map["left_eye"]) + calculate_ear(pts_map["right_eye"]))

        entries.append(
            {
                "img_path": full_path,
                "rel_name": rel_name,
                "w": w,
                "h": h,
                "left_eye": pts_map["left_eye"],
                "right_eye": pts_map["right_eye"],
                "mouth": pts_map["mouth"],
                "ear": ear_val,
                "mar": mar_val,
                "is_drowsy_folder": is_drowsy_folder,
                "dataset": dataset_tag,
            }
        )
    return entries


def build_yolo_line(w: int, h: int, left_eye: np.ndarray, right_eye: np.ndarray, mouth: np.ndarray) -> str:
    """Tạo chuỗi nhãn chuẩn YOLOv8-Pose (16 keypoints) từ tọa độ điểm mốc."""
    all_pts = np.vstack([left_eye, right_eye, mouth])
    xs = np.clip(all_pts[:, 0] / w, 0.0, 1.0)
    ys = np.clip(all_pts[:, 1] / h, 0.0, 1.0)

    # Ước lượng khung khuôn mặt (Bounding Box) bao quanh mắt + miệng với lề chuẩn của khuôn mặt
    eye_mouth_h = max(0.05, float(np.max(ys) - np.min(ys)))
    eye_mouth_w = max(0.05, float(np.max(xs) - np.min(xs)))

    cx = float(0.5 * (np.min(xs) + np.max(xs)))
    cy = float(0.5 * (np.min(ys) + np.max(ys)) - 0.04 * eye_mouth_h)
    bw = min(1.0, eye_mouth_w * 1.65)
    bh = min(1.0, eye_mouth_h * 2.15)

    cx = min(max(bw / 2.0, cx), 1.0 - bw / 2.0)
    cy = min(max(bh / 2.0, cy), 1.0 - bh / 2.0)

    kpt_tokens = []
    for kx, ky in zip(xs, ys):
        kpt_tokens.append(f"{float(kx):.6f} {float(ky):.6f} 2")

    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kpt_tokens)


def prepare_unified_dataset(
    yawdd_xml: str,
    yawdd_root: str,
    ddd_xml: str,
    ddd_root: str,
    out_dir: str,
    eye_clf: EyeStateClassifier,
    samples_per_group: int = 400,
    seed: int = 42,
) -> Tuple[str, dict]:
    """
    Xây dựng tập dữ liệu hợp nhất (YawDD + DDD) đã cân bằng phân phối và sửa nhãn mí mắt nhắm:
      1. Nhóm YawDD - Đang ngáp (MAR >= 0.50)
      2. Nhóm YawDD - Bình thường / Nói chuyện / Quay đầu (MAR < 0.50)
      3. Nhóm DDD - Buồn ngủ / Nhắm mắt (Drowsy + sửa nhãn mí mắt bằng EyeStateClassifier)
      4. Nhóm DDD - Tỉnh táo (Non Drowsy)
    """
    rng = random.Random(seed)
    print("[1/3] Dang doc du lieu tu CVAT_auto_annotations.xml (YawDD) va ddd_annotations.xml (DDD)...")
    yawdd_all = extract_xml_entries(yawdd_xml, yawdd_root, "yawdd")
    ddd_all = extract_xml_entries(ddd_xml, ddd_root, "ddd")

    yawdd_yawn = [e for e in yawdd_all if e["mar"] >= 0.48]
    yawdd_normal = [e for e in yawdd_all if e["mar"] < 0.48]
    ddd_drowsy = [e for e in ddd_all if e["is_drowsy_folder"]]
    ddd_alert = [e for e in ddd_all if not e["is_drowsy_folder"]]

    print(
        f"  -> YawDD (Ngap: {len(yawdd_yawn)}, Binh thuong: {len(yawdd_normal)}) | "
        f"DDD (Drowsy: {len(ddd_drowsy)}, Non-Drowsy: {len(ddd_alert)})"
    )

    rng.shuffle(yawdd_yawn)
    rng.shuffle(yawdd_normal)
    rng.shuffle(ddd_drowsy)
    rng.shuffle(ddd_alert)

    selected = (
        yawdd_yawn[:samples_per_group]
        + yawdd_normal[:samples_per_group]
        + ddd_drowsy[:samples_per_group]
        + ddd_alert[:samples_per_group]
    )
    rng.shuffle(selected)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    corrected_eyes_count = 0
    valid_written = 0
    val_cutoff = int(len(selected) * 0.85)

    print(f"[2/3] Dang lam sach nhan (sua loi ho mi mat MediaPipe bang EyeCNN) cho {len(selected)} anh...")
    for idx, item in enumerate(selected):
        img_path = item["img_path"]
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        left_eye = item["left_eye"].copy()
        right_eye = item["right_eye"].copy()
        mouth = item["mouth"].copy()

        # Kiểm tra từng mắt bằng mạng EyeStateClassifier (98.5% acc trên MRL Eye Dataset)
        if eye_clf.available:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            p_l = eye_clf.crop_eye_from_landmarks(gray, left_eye)
            p_r = eye_clf.crop_eye_from_landmarks(gray, right_eye)

            thresh_prob = 0.52 if item["is_drowsy_folder"] else 0.72
            pinched_any = False

            if p_l is not None:
                with torch.no_grad():
                    t_l = torch.from_numpy(p_l).unsqueeze(0).unsqueeze(0)
                    prob_l = float(torch.softmax(eye_clf.model(t_l), dim=1)[0, 1].item())
                if prob_l >= thresh_prob and calculate_ear(left_eye) > 0.14:
                    left_eye = pinch_eyelid_points(left_eye, pinch_ratio=0.64)
                    pinched_any = True

            if p_r is not None:
                with torch.no_grad():
                    t_r = torch.from_numpy(p_r).unsqueeze(0).unsqueeze(0)
                    prob_r = float(torch.softmax(eye_clf.model(t_r), dim=1)[0, 1].item())
                if prob_r >= thresh_prob and calculate_ear(right_eye) > 0.14:
                    right_eye = pinch_eyelid_points(right_eye, pinch_ratio=0.64)
                    pinched_any = True

            if pinched_any:
                corrected_eyes_count += 1

        split = "train" if idx < val_cutoff else "val"
        ext = os.path.splitext(img_path)[1]
        base_id = f"{item['dataset']}_{idx:05d}"
        dst_img = os.path.join(out_dir, "images", split, base_id + ext)
        dst_lbl = os.path.join(out_dir, "labels", split, base_id + ".txt")

        shutil.copy2(img_path, dst_img)
        yolo_line = build_yolo_line(w, h, left_eye, right_eye, mouth)
        with open(dst_lbl, "w", encoding="utf-8") as lf:
            lf.write(yolo_line + "\n")
        valid_written += 1

    yaml_path = os.path.join(out_dir, "data.yaml")
    out_dir_Clean = out_dir.replace("\\", "/")
    yaml_content = (
        f"path: {out_dir_Clean}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"kpt_shape: [16, 3]\n"
        f"flip_idx: [9, 8, 7, 6, 11, 10, 3, 2, 1, 0, 5, 4, 14, 13, 12, 15]\n"
        f"names:\n"
        f"  0: face\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    stats = {
        "total_written": valid_written,
        "corrected_eyes": corrected_eyes_count,
    }
    print(
        f"  -> Da tao xong dataset hop nhat tai {out_dir}: {valid_written} anh "
        f"(Da sua nhan mi mat nham cho {corrected_eyes_count} anh!)."
    )
    return yaml_path, stats


def main():
    parser = argparse.ArgumentParser(description="Hợp nhất dữ liệu YawDD + DDD và huấn luyện 1 model dms_unified.pt duy nhất")
    parser.add_argument("--samples-per-group", type=int, default=200, help="Số ảnh mỗi phân nhóm (x4 nhóm)")
    parser.add_argument("--epochs", type=int, default=4, help="Số epoch fine-tune hợp nhất")
    parser.add_argument("--imgsz", type=int, default=416, help="Kích thước ảnh huấn luyện")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--out-model", type=str, default=os.path.join(ROOT_DIR, "models", "dms_unified.pt"))
    args = parser.parse_args()

    eye_clf_path = os.path.join(ROOT_DIR, "models", "eye_classifier.pt")
    eye_clf = EyeStateClassifier(eye_clf_path)

    dataset_dir = r"D:\DMS_Dataset\YOLO_Unified_Pose"
    yaml_path, _ = prepare_unified_dataset(
        yawdd_xml=r"D:\DMS_Dataset\CVAT_auto_annotations.xml",
        yawdd_root=r"D:\DMS_Dataset\yawdd_frames",
        ddd_xml=r"D:\DMS_Dataset\ddd_annotations.xml",
        ddd_root=r"D:\DMS_Dataset\driver_drowsiness_dataset",
        out_dir=dataset_dir,
        eye_clf=eye_clf,
        samples_per_group=args.samples_per_group,
    )

    init_pt = os.path.join(ROOT_DIR, "models", "yawdd.pt")

    print("[3/3] Bat dau Fine-tune Model Hop Nhat (dms_unified.pt) tren tap YawDD + DDD da sua nhan...")
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        torch.set_num_threads(4)

    model = YOLO(init_pt)
    run_dir = os.path.join(dataset_dir, "runs")
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu" if not torch.cuda.is_available() else "0",
        workers=2,
        freeze=12,           # Giữ nguyên Backbone + Neck đầu của yawdd.pt, tinh chỉnh nhánh Pose Head học thêm DDD + mắt nhắm
        lr0=0.0008,
        lrf=0.1,
        optimizer="AdamW",
        fliplr=0.5,
        mosaic=0.3,
        project=run_dir,
        name="unified_pose",
        exist_ok=True,
        plots=False,
        verbose=True,
    )

    best_pt = os.path.join(run_dir, "unified_pose", "weights", "best.pt")
    if not os.path.exists(best_pt):
        best_pt = os.path.join(run_dir, "unified_pose", "weights", "last.pt")

    # Đóng gói cả trọng số EyeStateNet (280KB) vào bên trong file dms_unified.pt
    # để toàn bộ hệ thống có thể nạp từ ĐÚNG 1 FILE DUY NHẤT!
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    if os.path.exists(eye_clf_path):
        eye_ckpt = torch.load(eye_clf_path, map_location="cpu", weights_only=True)
        ckpt["eye_classifier_state_dict"] = (
            eye_ckpt["model_state_dict"] if isinstance(eye_ckpt, dict) and "model_state_dict" in eye_ckpt else eye_ckpt
        )
        ckpt["eye_classifier_val_acc"] = eye_ckpt.get("val_acc", 0.98) if isinstance(eye_ckpt, dict) else 0.98

    os.makedirs(os.path.dirname(os.path.abspath(args.out_model)), exist_ok=True)
    torch.save(ckpt, args.out_model)
    print(f"\n[HOAN TAT] Da luu Model Hop Nhat Duy Nhat tai: {args.out_model}")


if __name__ == "__main__":
    main()
