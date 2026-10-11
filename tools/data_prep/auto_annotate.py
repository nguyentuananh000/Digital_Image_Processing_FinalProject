import argparse
import glob
import os
import cv2
import mediapipe as mp


# Thứ tự chuẩn 16 điểm mốc: 6 mắt trái + 6 mắt phải + 4 miệng
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [78, 13, 308, 14]
# 2 điểm mở rộng (Đỉnh mũi + Đỉnh cằm) cho bài toán Head Pose 3D (18 điểm mốc)
POSE_EXTRA_IDX = [1, 152]


def generate_annotations(image_dir: str, output_xml: str, yolo_labels_dir: str = None, num_kpts: int = 16):
    """
    Tự động quét ảnh và gán nhãn điểm mốc khuôn mặt bằng MediaPipe FaceMesh.
    - Xuất file XML chuẩn CVAT để kiểm tra/chỉnh sửa thủ công.
    - Tùy chọn xuất trực tiếp nhãn định dạng YOLOv8-Pose (.txt) nếu truyền `yolo_labels_dir`.
    - Hỗ trợ 16 keypoints (Mắt + Miệng) hoặc 18 keypoints (+ Mũi, Cằm cho Head Pose 3D).
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    )

    xml_content = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>"]

    exts = ("*.png", "*.jpg", "*.jpeg")
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))

    print(f"Bat dau quet va cham diem tu dong ({num_kpts} keypoints) cho {len(image_files)} anh...")

    if yolo_labels_dir:
        os.makedirs(yolo_labels_dir, exist_ok=True)

    annotated_count = 0
    for idx, img_path in enumerate(image_files):
        rel_path = os.path.relpath(img_path, image_dir).replace("\\", "/")
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape

        xml_content.append(f'  <image id="{idx}" name="{rel_path}" width="{w}" height="{h}">')

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            annotated_count += 1
            face_lm = results.multi_face_landmarks[0].landmark
            lm = [(int(pt.x * w), int(pt.y * h)) for pt in face_lm]

            left_pts = ";".join([f"{lm[i][0]},{lm[i][1]}" for i in LEFT_EYE_IDX])
            right_pts = ";".join([f"{lm[i][0]},{lm[i][1]}" for i in RIGHT_EYE_IDX])
            mouth_pts = ";".join([f"{lm[i][0]},{lm[i][1]}" for i in MOUTH_IDX])

            xml_content.append(f'    <points label="left_eye" source="auto" occluded="0" points="{left_pts}"/>')
            xml_content.append(f'    <points label="right_eye" source="auto" occluded="0" points="{right_pts}"/>')
            xml_content.append(f'    <points label="mouth" source="auto" occluded="0" points="{mouth_pts}"/>')

            if num_kpts == 18:
                pose_pts = ";".join([f"{lm[i][0]},{lm[i][1]}" for i in POSE_EXTRA_IDX])
                xml_content.append(f'    <points label="head_pose" source="auto" occluded="0" points="{pose_pts}"/>')

            # Xuất trực tiếp file .txt chuẩn YOLOv8-Pose nếu được yêu cầu
            if yolo_labels_dir:
                xs = [min(max(0.0, pt.x), 1.0) for pt in face_lm]
                ys = [min(max(0.0, pt.y), 1.0) for pt in face_lm]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                bw = min(1.0, (x_max - x_min) * 1.08)
                bh = min(1.0, (y_max - y_min) * 1.08)

                all_indices = LEFT_EYE_IDX + RIGHT_EYE_IDX + MOUTH_IDX
                if num_kpts == 18:
                    all_indices += POSE_EXTRA_IDX

                kpt_tokens = []
                for k_idx in all_indices:
                    kx = min(max(0.0, face_lm[k_idx].x), 1.0)
                    ky = min(max(0.0, face_lm[k_idx].y), 1.0)
                    kpt_tokens.append(f"{kx:.6f} {ky:.6f} 2")

                yolo_line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kpt_tokens)
                base_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
                with open(os.path.join(yolo_labels_dir, base_name), "w", encoding="utf-8") as lf:
                    lf.write(yolo_line + "\n")

        xml_content.append("  </image>")

    xml_content.append("</annotations>")

    os.makedirs(os.path.dirname(os.path.abspath(output_xml)), exist_ok=True)
    with open(output_xml, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_content))

    print(f"HOAN TAT! Da gan nhan {annotated_count}/{len(image_files)} anh. Luu XML tai: {output_xml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tự động gán nhãn điểm mốc khuôn mặt cho CVAT / YOLOv8-Pose")
    parser.add_argument("--image-dir", type=str, default=r"D:\DMS_Dataset\driver_drowsiness_dataset")
    parser.add_argument("--output-xml", type=str, default=r"D:\DMS_Dataset\ddd_annotations.xml")
    parser.add_argument("--yolo-dir", type=str, default=None, help="Thư mục xuất trực tiếp nhãn .txt YOLOv8-Pose")
    parser.add_argument("--keypoints", type=int, choices=[16, 18], default=16, help="Số điểm mốc (16 hoặc 18)")
    args = parser.parse_args()

    generate_annotations(args.image_dir, args.output_xml, args.yolo_dir, args.keypoints)