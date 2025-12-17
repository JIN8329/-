"""
4단계: 학습된 LSTM 모델을 이용하여 새로운 영상에서
      낙상 여부를 실시간 판단하는 코드 (수정 완료)
"""

from ultralytics import YOLO
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# --------------------------
# YOLO 포즈 모델 로드 + GPU
# --------------------------
model_pose = YOLO("yolov8n-pose.pt")
model_pose.to("cuda:0")

model_lstm = load_model("fall_lstm_model.h5")

# --------------------------
# 설정값
# --------------------------
window = 30
sequence_buffer = []

# COCO keypoint index
L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12


def center(p1, p2):
    return (p1 + p2) / 2.0


def angle(p_up, p_down):
    dx = p_down[0] - p_up[0]
    dy = p_down[1] - p_up[1]
    return np.arctan2(dy, dx)


# --------------------------
# 테스트 영상 로드
# --------------------------
video = cv2.VideoCapture(
    r"C:\Users\HaJinYeong\Desktop\테스트 영상\시니어 이상행동 영상\Validation\동영상\Abnormal_Behavior_Falldown\[원천]inside_H12H22H31\H12H22H31\FD_In_H12H22H31_0009_20210112_18.mp4"
)

prev_valid_feature = None

while True:
    ret, frame = video.read()
    if not ret:
        break

    # -----------------------------
    # YOLO Pose
    # -----------------------------
    results = model_pose(frame, verbose=False)
    kp = results[0].keypoints

    feature = None

    if kp is not None and kp.xy is not None and len(kp.xy) > 0:
        pts = kp.xy[0].cpu().numpy()  # (17, 2)

        # 중앙점
        sc = center(pts[L_SHOULDER], pts[R_SHOULDER])
        hc = center(pts[L_HIP], pts[R_HIP])
        wc = center(sc, hc)

        # 기울기
        ang1 = angle(sc, wc)
        ang2 = angle(sc, hc)

        feature = np.concatenate([sc, hc, wc, np.array([ang1, ang2])])

    # -----------------------------
    # 관절 검출 실패 시 보간
    # -----------------------------
    if feature is None:
        print("⚠ 관절 검출 실패 → 이전 Feature 사용")
        feature = prev_valid_feature if prev_valid_feature is not None else np.zeros(8)

    prev_valid_feature = feature

    # -----------------------------
    # LSTM 입력 버퍼
    # -----------------------------
    sequence_buffer.append(feature)

    if len(sequence_buffer) >= window:
        X = np.array(sequence_buffer[-window:]).reshape(1, window, 8)
        pred = model_lstm.predict(X, verbose=False)[0][0]

        label = "FALL!!" if pred > 0.5 else "Normal"
        color = (0, 0, 255) if pred > 0.5 else (0, 255, 0)

        cv2.putText(frame, label, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # 화면 출력
    scaled = cv2.resize(frame, None, fx=0.5, fy=0.5)
    cv2.imshow("Test", scaled)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
