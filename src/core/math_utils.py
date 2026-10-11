import cv2
import math
import numpy as np


def apply_clahe(image):
    """Cân bằng sáng cục bộ (CLAHE) trên kênh L của không gian màu LAB giúp cải thiện ảnh ngược sáng/thiếu sáng."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def euclidean_dist(p1, p2):
    """Tính khoảng cách Euclid giữa 2 điểm 2D (x, y)."""
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def calculate_ear(eye_points):
    """
    Tính chỉ số EAR kết hợp (Hybrid Line + Shoelace Polygon Area EAR) cho 6 điểm của mắt:
    - Kết hợp khoảng cách dọc (v1 + v2) / (2 * h) truyền thống với diện tích đa giác 6 điểm mắt (Shoelace formula)
      nhằm giảm sai số khi 1 trong 4 điểm mí mắt bị lệch 1-2 pixels.
    """
    if eye_points is None or len(eye_points) != 6:
        return 0.0

    for pt in eye_points:
        if pt[0] <= 0 and pt[1] <= 0:
            return 0.0

    v1 = euclidean_dist(eye_points[1], eye_points[5])
    v2 = euclidean_dist(eye_points[2], eye_points[4])
    h = euclidean_dist(eye_points[0], eye_points[3])

    if h <= 1e-6:
        return 0.0

    classic_ear = (v1 + v2) / (2.0 * h)

    # Tính diện tích đa giác 6 điểm mắt bằng công thức Shoelace chuẩn hóa theo bình phương chiều ngang mắt
    pts = np.asarray(eye_points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    poly_area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    area_ear = (poly_area / (h * h)) * 1.35

    # Pha trộn 70% EAR cổ điển + 30% Area-EAR giúp chỉ số ổn định hơn trước nhiễu điểm mốc
    return 0.70 * classic_ear + 0.30 * area_ear


def combine_bino_ear(ear_left: float, ear_right: float, yaw: float = 0.0) -> float:
    """
    Hợp nhất EAR của 2 mắt có trọng số bất đối xứng (Asymmetric Eye Weighting):
    Khi tài xế quay đầu sang một bên (|Yaw| > 10 độ), mắt ở xa camera bị hẹp góc chiếu 2D,
    do đó ưu tiên trọng số cao hơn (75%) cho con mắt mở rõ hơn (mắt gần camera).
    """
    if ear_left <= 0:
        return ear_right
    if ear_right <= 0:
        return ear_left

    if abs(yaw) > 10.0:
        high_ear = max(ear_left, ear_right)
        low_ear = min(ear_left, ear_right)
        return 0.75 * high_ear + 0.25 * low_ear

    return (ear_left + ear_right) / 2.0


def calculate_mar(mouth_points):
    """
    Tính chỉ số MAR (Mouth Aspect Ratio) cho 4 điểm của miệng.
    Thứ tự điểm: 0: Khóe trái (78) | 1: Môi trên (13) | 2: Khóe phải (308) | 3: Môi dưới (14)
    """
    if mouth_points is None or len(mouth_points) != 4:
        return 0.0

    for pt in mouth_points:
        if pt[0] <= 0 and pt[1] <= 0:
            return 0.0

    v = euclidean_dist(mouth_points[1], mouth_points[3])
    h = euclidean_dist(mouth_points[0], mouth_points[2])

    return v / h if h > 1e-6 else 0.0


def get_head_pose(shape, image_pts):
    """
    Tính góc nghiêng đầu 3D (Pitch, Yaw, Roll) bằng thuật toán solvePnP với 6 điểm mốc chuẩn.
    Thứ tự 6 điểm: [Mũi, Cằm, Khóe mắt trái, Khóe mắt phải, Khóe miệng trái, Khóe miệng phải]
    """
    model_pts = np.array([
        (0.0, 0.0, 0.0),             # Mũi (Nose tip)
        (0.0, -330.0, -65.0),        # Cằm (Chin)
        (-225.0, 170.0, -135.0),     # Khóe mắt trái
        (225.0, 170.0, -135.0),      # Khóe mắt phải
        (-150.0, -150.0, -125.0),    # Khóe miệng trái
        (150.0, -150.0, -125.0)      # Khóe miệng phải
    ], dtype="double")

    focal_length = shape[1]
    center = (shape[1] / 2.0, shape[0] / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_pts, image_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    # cv2.decomposeProjectionMatrix trả về góc Euler đã ở đơn vị Độ (Degrees)
    raw_pitch = float(euler_angles[0, 0])
    raw_yaw = float(euler_angles[1, 0])
    raw_roll = float(euler_angles[2, 0])

    # Chuẩn hóa góc Pitch về quanh 0 độ (nhìn thẳng ~ 0 độ, cúi đầu gục xuống < 0 độ)
    if raw_pitch > 90.0:
        pitch = raw_pitch - 180.0
    elif raw_pitch < -90.0:
        pitch = raw_pitch + 180.0
    else:
        pitch = raw_pitch

    yaw = raw_yaw
    roll = raw_roll
    return pitch, yaw, roll


def estimate_pose_from_16kpts(shape, kpts_16, bbox=None):
    """
    Ước lượng góc đầu (Pitch, Yaw, Roll) từ 16 điểm mốc của YOLOv8-Pose (6 mắt trái, 6 mắt phải, 4 miệng)
    kết hợp với khung khuôn mặt (bbox), giúp mô hình YOLO cũng phát hiện được hành vi gục đầu / quay đầu.
    """
    key_indices = [0, 3, 6, 9, 12, 14]
    for idx in key_indices:
        if kpts_16[idx][0] <= 0 or kpts_16[idx][1] <= 0:
            return None, None, None

    le_outer = kpts_16[0]
    le_inner = kpts_16[3]
    re_inner = kpts_16[6]
    re_outer = kpts_16[9]
    m_center = (kpts_16[13] + kpts_16[15]) / 2.0

    # 1. Tính Roll (nghiêng đầu trái/phải) từ đường nối 2 mắt
    dx = float(re_outer[0] - le_outer[0])
    dy = float(re_outer[1] - le_outer[1])
    roll = math.degrees(math.atan2(dy, dx)) if abs(dx) > 1e-6 else 0.0

    # 2. Tính Yaw (quay đầu trái/phải) từ độ lệch tâm giữa mắt và miệng
    eyes_mid_x = (le_inner[0] + re_inner[0]) / 2.0
    eye_span = euclidean_dist(le_outer, re_outer)
    if eye_span < 1e-6:
        return 0.0, 0.0, roll

    left_span = abs(eyes_mid_x - le_outer[0])
    right_span = abs(re_outer[0] - eyes_mid_x)
    yaw_ratio = (right_span - left_span) / (left_span + right_span + 1e-6)
    yaw = yaw_ratio * 90.0

    # 3. Ước lượng Pitch (cúi/ngửa đầu)
    eyes_mid_y = (le_outer[1] + re_outer[1] + le_inner[1] + re_inner[1]) / 4.0
    eye_to_mouth_y = float(m_center[1] - eyes_mid_y)
    vertical_ratio = eye_to_mouth_y / eye_span

    if bbox is not None and (bbox[3] - bbox[1]) > 10:
        face_h = float(bbox[3] - bbox[1])
        chin_margin = float(bbox[3] - m_center[1]) / face_h
        pitch = (vertical_ratio - 0.65) * 45.0 + (chin_margin - 0.22) * 60.0
    else:
        pitch = (vertical_ratio - 0.65) * 60.0

    return pitch, yaw, roll
