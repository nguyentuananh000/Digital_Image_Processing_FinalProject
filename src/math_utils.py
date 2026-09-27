import cv2
import math
import numpy as np

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def calculate_ear(eye_points):
    v1 = math.dist(eye_points[1], eye_points[5])
    v2 = math.dist(eye_points[2], eye_points[4])
    h = math.dist(eye_points[0], eye_points[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0

def calculate_mar(mouth_points):
    v = math.dist(mouth_points[1], mouth_points[3])
    h = math.dist(mouth_points[0], mouth_points[2])
    return v / h if h != 0 else 0

def get_head_pose(shape, image_pts):
    model_pts = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ])
    focal_length = shape[1]
    center = (shape[1]/2, shape[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4,1))
    
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_pts, image_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    
    pitch, yaw, roll = [math.degrees(e[0]) for e in eulerAngles]
    return pitch, yaw, roll