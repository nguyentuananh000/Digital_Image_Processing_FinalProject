import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import sys
import os

# Xử lý đường dẫn để import an toàn trong thư mục src
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Nhập các module đã tách
from camera import WebcamStream
from math_utils import apply_clahe, calculate_ear, calculate_mar, get_head_pose
from audio_manager import AudioManager

def main():
    # Cấu hình hệ thống mặc định
    EAR_THRESH, MAR_THRESH, PITCH_THRESH = 0.22, 0.5, -15.0   
    SLEEP_TIME, NODDING_TIME, YAWN_TIME = 2.0, 1.5, 2.0

    ear_history = deque(maxlen=5)
    mar_history = deque(maxlen=5)

    is_calibrating = True
    calib_start_time = None
    calib_duration = 5.0 
    calib_ear_data, calib_mar_data = [], []

    # Trỏ động đến thư mục assets
    sound_path = os.path.join(current_dir, '..', 'assets', 'alarm.wav')
    audio = AudioManager(sound_path)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    print("Dang khoi dong he thong DMS...")
    vs = WebcamStream(src=0).start()
    time.sleep(1.0) 

    sleep_start = nod_start = yawn_start = None
    prev_time = 0

    while True:
        success, frame = vs.read()
        if not success: break

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        frame = apply_clahe(frame)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

                left_eye = [lm[33], lm[160], lm[158], lm[133], lm[153], lm[144]]
                right_eye = [lm[362], lm[385], lm[387], lm[263], lm[373], lm[380]]
                mouth = [lm[78], lm[13], lm[308], lm[14]]
                
                ear_history.append((calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0)
                mar_history.append(calculate_mar(mouth))
                ear, mar = sum(ear_history)/len(ear_history), sum(mar_history)/len(mar_history)

                image_pts = np.array([lm[1], lm[152], lm[33], lm[263], lm[61], lm[291]], dtype="double")
                pitch, yaw, roll = get_head_pose(frame.shape, image_pts)

                if is_calibrating:
                    if calib_start_time is None: calib_start_time = time.time()
                    elapsed = time.time() - calib_start_time
                    time_left = max(0, int(calib_duration - elapsed))
                    
                    if elapsed < calib_duration:
                        calib_ear_data.append(ear)
                        calib_mar_data.append(mar)
                        cv2.putText(frame, f"DANG HIEU CHUAN... {time_left}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                        cv2.putText(frame, "Nhin thang, mo mat binh thuong", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        if calib_ear_data: EAR_THRESH = (sum(calib_ear_data) / len(calib_ear_data)) * 0.75
                        if calib_mar_data: MAR_THRESH = (sum(calib_mar_data) / len(calib_mar_data)) + 0.25
                        is_calibrating = False
                        print(f"Hieu chuan xong: EAR {EAR_THRESH:.2f}, MAR {MAR_THRESH:.2f}")

                else:
                    status_text, status_color, trigger_alarm = "TINH TAO", (0, 255, 0), False

                    if ear < EAR_THRESH:
                        if sleep_start is None: sleep_start = time.time()
                        elif time.time() - sleep_start > SLEEP_TIME:
                            status_text, status_color, trigger_alarm = "CANH BAO: NGU GAT!", (0, 0, 255), True
                    else: sleep_start = None

                    if pitch < PITCH_THRESH:
                        if nod_start is None: nod_start = time.time()
                        elif time.time() - nod_start > NODDING_TIME:
                            status_text, status_color, trigger_alarm = "CANH BAO: GUC DAU!", (0, 165, 255), True
                    else: nod_start = None

                    if mar > MAR_THRESH:
                        if yawn_start is None: yawn_start = time.time()
                        elif time.time() - yawn_start > YAWN_TIME:
                            cv2.putText(frame, "Luu y: Tai xe dang Ngap", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    else: yawn_start = None

                    audio.play() if trigger_alarm else audio.stop()
                    cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

                cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 255), 1)
                cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 255), 1)
                cv2.putText(frame, f"Pitch: {int(pitch)}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"EAR: {ear:.2f}", (w - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"FPS: {int(fps)}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow('He thong giam sat tai xe', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()