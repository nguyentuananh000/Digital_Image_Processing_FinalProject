import argparse
import os
import cv2
import albumentations as A


def build_augmentation_pipeline(with_keypoints: bool = False):
    """
    Kịch bản tăng cường dữ liệu (Data Augmentation) an toàn cho bài toán nhận diện điểm mốc mắt/miệng:
    - Giảm kích thước CoarseDropout (tối đa 24x24) để không xóa mất toàn bộ vùng mắt khi huấn luyện Pose.
    - Hỗ trợ biến đổi đồng thời tọa độ Keypoints nếu truyền `with_keypoints=True`.
    """
    transforms = [
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.7),
        A.RandomGamma(gamma_limit=(70, 140), p=0.5),
        A.MotionBlur(blur_limit=9, p=0.4),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(12, 24),
            hole_width_range=(12, 24),
            fill=0,
            p=0.4,
        ),
    ]
    if with_keypoints:
        return A.Compose(
            transforms,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )
    return A.Compose(transforms)


def augment_image(image_path: str, output_dir: str, num_versions: int = 5):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[LOI] Khong tim thay anh tai: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = build_augmentation_pipeline(with_keypoints=False)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"Dang tao {num_versions} phien ban tang cuong tu anh: {image_path}...")
    for i in range(num_versions):
        augmented = transform(image=image_rgb)
        aug_img_bgr = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
        out_path = os.path.join(output_dir, f"{base_name}_aug_{i + 1}.jpg")
        cv2.imwrite(out_path, aug_img_bgr)
        print(f"  -> Da luu: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tăng cường dữ liệu ảnh cho hệ thống DMS")
    parser.add_argument("--image", type=str, required=False, default="test_image.jpg", help="Đường dẫn ảnh đầu vào")
    parser.add_argument("--output-dir", type=str, default="augmented_output", help="Thư mục lưu ảnh sau khi tăng cường")
    parser.add_argument("--count", type=int, default=5, help="Số lượng phiên bản tạo ra")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Vui long truyen duong dan anh hop le qua --image (Hien tai khong tim thay '{args.image}').")
    else:
        augment_image(args.image, args.output_dir, args.count)