import os
import glob
import shutil

def separate_images(mixed_dir, target_image_dir):
    # Quét tìm toàn bộ file ảnh .jpg trong thư mục đang bị lẫn lộn
    search_pattern = os.path.join(mixed_dir, "**", "*.jpg")
    image_files = glob.glob(search_pattern, recursive=True)
    
    if not image_files:
        print("Khong tim thay anh nao trong thu muc nay!")
        return
        
    print(f"Tim thay {len(image_files)} anh. Dang tien hanh di chuyen...")
    
    moved_count = 0
    for img_path in image_files:
        # 1. Trích xuất đường dẫn tương đối (Ví dụ: "Male/Yawning")
        img_dir = os.path.dirname(img_path)
        rel_path = os.path.relpath(img_dir, mixed_dir)
        
        # 2. Tạo thư mục con tương ứng ở thư mục đích
        dest_dir = os.path.join(target_image_dir, rel_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            
        # 3. Di chuyển file ảnh (cắt từ thư mục cũ sang thư mục mới)
        dest_path = os.path.join(dest_dir, os.path.basename(img_path))
        shutil.move(img_path, dest_path)
        moved_count += 1
        
    print(f"HOAN TAT! Da di chuyen {moved_count} anh sang {target_image_dir}.")
    print("Thu muc goc hien tai da sach se, chi con lai video.")

if __name__ == "__main__":
    # CHÚ Ý: Cập nhật đường dẫn của bạn
    # 1. Thư mục đang bị lẫn lộn ảnh và video
    MIXED_DIR = r"D:\DMS_Dataset\yawdd_dataset" 
    
    # 2. Thư mục mới CHỈ CHỨA ẢNH
    TARGET_IMAGE_DIR = r"D:\DMS_Dataset\yawdd_frames"
    
    separate_images(MIXED_DIR, TARGET_IMAGE_DIR)