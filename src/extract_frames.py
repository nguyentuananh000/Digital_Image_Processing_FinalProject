import cv2
import os
import glob

def extract_frames(video_path, base_input_dir, base_output_dir, frames_per_sec=2):
    # 1. Trích xuất đường dẫn thư mục con chứa video hiện tại
    video_dir = os.path.dirname(video_path)
    
    # 2. Lấy cấu trúc thư mục tương đối (Ví dụ: "nam/ngap")
    rel_path = os.path.relpath(video_dir, base_input_dir)
    
    # 3. Tạo thư mục con tương ứng ở thư mục đầu ra
    target_output_dir = os.path.join(base_output_dir, rel_path)
    if not os.path.exists(target_output_dir):
        os.makedirs(target_output_dir)
        
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if fps == 0: 
        fps = 30
        
    interval = max(1, int(fps / frames_per_sec))
    count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % interval == 0:
            image_name = f"{video_name}_frame_{saved_count:04d}.jpg"
            image_path = os.path.join(target_output_dir, image_name)
            cv2.imwrite(image_path, frame)
            saved_count += 1
            
        count += 1
        
    cap.release()
    print(f"[OK] {video_name} -> Luu vao thu muc: {rel_path} ({saved_count} anh)")

if __name__ == "__main__":
    # CHÚ Ý: Cập nhật đường dẫn của bạn vào đây
    INPUT_VIDEO_DIR = r"D:\DMS_Dataset\yawdd_dataset" 
    OUTPUT_IMAGE_DIR = r"D:\DMS_Dataset\yawdd_dataset"
    
    # Quét toàn bộ video .avi trong thư mục gốc và các thư mục con
    video_files = glob.glob(os.path.join(INPUT_VIDEO_DIR, "**", "*.avi"), recursive=True)
    
    print(f"Tim thay {len(video_files)} video. Dang bat dau xu ly...")
    
    for video_file in video_files:
        # Truyền thêm tham số INPUT_VIDEO_DIR và OUTPUT_IMAGE_DIR để đối chiếu cấu trúc
        extract_frames(video_file, INPUT_VIDEO_DIR, OUTPUT_IMAGE_DIR, frames_per_sec=2)
        
    print("HOAN TAT! Toan bo cau truc thu muc da duoc giu nguyen.")