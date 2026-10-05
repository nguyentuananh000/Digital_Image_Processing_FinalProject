import cv2
import albumentations as A
import os

def augment_image(image_path, output_dir, num_versions=5):
    # Đọc ảnh gốc
    image = cv2.imread(image_path)
    if image is None:
        print("Khong tim thay anh!")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Khai báo kịch bản "làm xấu" ảnh
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7), # Giả lập ngược nắng hoặc trời tối
        A.MotionBlur(blur_limit=15, p=0.5), # Giả lập camera rung lắc do xe chạy qua ổ gà
        A.ISONoise(p=0.5), # Nhiễu hột camera ban đêm
        # CoarseDropout: Cắt các mảng pixel ngẫu nhiên để giả lập tài xế đeo kính râm, khẩu trang hoặc bị che khuất
        A.CoarseDropout(max_holes=3, max_height=80, max_width=80, fill_value=0, p=0.5), 
    ])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Dang tao {num_versions} phien ban tu anh goc...")
    for i in range(num_versions):
        # Áp dụng hiệu ứng ngẫu nhiên
        augmented = transform(image=image)
        aug_img = augmented['image']
        
        # Lưu lại
        aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(output_dir, f"aug_version_{i+1}.jpg")
        cv2.imwrite(out_path, aug_img_bgr)
        print(f"Da luu: {out_path}")

if __name__ == "__main__":
    # CHÚ Ý: Đổi đường dẫn này thành một file ảnh bất kỳ có trên máy bạn để test
    SAMPLE_IMAGE = r"C:\Users\nguyentuananh\Pictures\hw1\anh tay.jpg" 
    OUTPUT_FOLDER = r"C:\Users\nguyentuananh\Pictures\hw1\images"
    
    # Tạo một file ảnh tạm thời để bạn test nếu chưa có ảnh
    if not os.path.exists(SAMPLE_IMAGE):
        dummy_img = cv2.imread("test_image.jpg") # Cần trỏ đến 1 ảnh mặt người có thật
        print("Vui long dat 1 buc anh mat nguoi ten 'test_image.jpg' vao thu muc de chay thu!")
    else:
        augment_image(SAMPLE_IMAGE, OUTPUT_FOLDER)