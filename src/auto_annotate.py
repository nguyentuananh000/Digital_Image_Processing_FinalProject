import cv2
import mediapipe as mp
import os
import glob

def generate_cvat_xml(image_dir, output_xml):
    # Khởi tạo MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Cấu trúc file XML chuẩn của CVAT
    xml_content = ['<?xml version="1.0" encoding="utf-8"?>', '<annotations>']

    # Tìm toàn bộ ảnh
    image_files = glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True)
    print(f"Bat dau quet va cham diem tu dong cho {len(image_files)} anh...")

    for idx, img_path in enumerate(image_files):
        # Lấy đường dẫn tương đối giống với cấu trúc bạn đã up lên CVAT
        rel_path = os.path.relpath(img_path, image_dir).replace('\\', '/')
        
        img = cv2.imread(img_path)
        if img is None: 
            continue
        h, w, _ = img.shape
        
        xml_content.append(f'  <image id="{idx}" name="{rel_path}" width="{w}" height="{h}">')
        
        # Đưa ảnh qua AI MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            lm = [(int(pt.x * w), int(pt.y * h)) for pt in results.multi_face_landmarks[0].landmark]
            
            # Lấy đúng index các điểm như trong code hệ thống của bạn
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]
            mouth_idx = [78, 13, 308, 14]
            
            # Nối tọa độ thành chuỗi "x1,y1;x2,y2;..."
            left_pts = ";".join([f"{lm[i][0]},{lm[i][1]}" for i in left_eye_idx])
            right_pts = ";".join([f"{lm[i][0]},{lm[i][1]}" for i in right_eye_idx])
            mouth_pts = ";".join([f"{lm[i][0]},{lm[i][1]}" for i in mouth_idx])
            
            # Ghi vào XML
            xml_content.append(f'    <points label="left_eye" source="auto" occluded="0" points="{left_pts}"/>')
            xml_content.append(f'    <points label="right_eye" source="auto" occluded="0" points="{right_pts}"/>')
            xml_content.append(f'    <points label="mouth" source="auto" occluded="0" points="{mouth_pts}"/>')
            
        xml_content.append('  </image>')

    xml_content.append('</annotations>')

    # Lưu file
    with open(output_xml, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_content))

    print(f"HOAN TAT! Da luu ket qua vao: {output_xml}")

if __name__ == "__main__":
    # Đảm bảo đường dẫn này trỏ tới thư mục ảnh đã được lọc
    IMAGE_DIR = r"D:\DMS_Dataset\driver_drowsiness_dataset"
    OUTPUT_XML = r"D:\DMS_Dataset\ddd_annotations.xml"
    
    generate_cvat_xml(IMAGE_DIR, OUTPUT_XML)