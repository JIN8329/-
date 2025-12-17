"""
1단계: 여러 영상에서 관절점을 추출하고
      허리 관절점과 기울기를 계산하여
      시계열(np.array) 형태로 저장하는 코드
"""

from ultralytics import YOLO
import numpy as np
import cv2
import os

# --------------------------
# YOLO 모델 로드 (Pose 모델)
# --------------------------
model = YOLO("yolov8n-pose.pt")
model.to("cuda:0")
print("YOLO device:", model.device)

# --------------------------
# 데이터 경로
# --------------------------
video_root = r"C:\Users\HaJinYeong\2025-2\videos"             # 여러 영상이 들어있는 폴더
save_root = r"C:\Users\HaJinYeong\2025-2\processed_sequences" # 시계열 저장 폴더
os.makedirs(save_root, exist_ok=True)

# --------------------------
# 관절 인덱스 (COCO 17 keypoints)
# --------------------------
L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12


def compute_center(p1, p2):
    """두 관절의 중앙점을 계산"""
    return (p1 + p2) / 2.0


def compute_angle(p_up, p_down):
    """두 점을 이용해 기울기(라디안) 계산"""
    dx = p_down[0] - p_up[0]
    dy = p_down[1] - p_up[1]
    return np.arctan2(dy, dx)


def interpolate_missing(sequence):
    """
    None으로 저장된 프레임을 이전/다음 값 평균으로 보간한다.
    sequence: 길이 T의 리스트, 각 요소는 (8,) 또는 None
    """

    seq = sequence.copy()

    # 우선 None이 아닌 인덱스를 찾음
    valid_indices = [i for i, v in enumerate(seq) if v is not None]

    if not valid_indices:
        return None  # 전 프레임 실패 → 사용 불가

    # 앞쪽 None 채우기
    first_valid = valid_indices[0]
    for i in range(0, first_valid):
        seq[i] = seq[first_valid]  # 동일 값으로 채움

    # 뒤쪽 None 채우기
    last_valid = valid_indices[-1]
    for i in range(last_valid + 1, len(seq)):
        seq[i] = seq[last_valid]

    # 중간 None 보간
    for i in range(1, len(seq) - 1):
        if seq[i] is None:
            # 좌우에서 가장 가까운 값 찾기
            left = i - 1
            while left >= 0 and seq[left] is None:
                left -= 1

            right = i + 1
            while right < len(seq) and seq[right] is None:
                right += 1

            seq[i] = (seq[left] + seq[right]) / 2.0

    return seq


# --------------------------
# 모든 영상 처리
# --------------------------
for file in os.listdir(video_root):
    if not file.endswith(".mp4"):
        continue

    video_path = os.path.join(video_root, file)
    cap = cv2.VideoCapture(video_path)

    sequence = []  # 각 프레임의 feature 또는 None 저장

    print(f">>> 처리 중: {file}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 추출
        results = model(frame, verbose=False)
        kp = results[0].keypoints

        # 검출 실패 → None 저장
        if kp is None or kp.xy is None or len(kp.xy) == 0:
            sequence.append(None)
            continue

        # 사람 한 명 기준 (0번)
        pts = kp.xy[0].cpu().numpy()

        # 중앙점 계산
        shoulder_center = compute_center(pts[L_SHOULDER], pts[R_SHOULDER])
        hip_center = compute_center(pts[L_HIP], pts[R_HIP])
        waist_center = compute_center(shoulder_center, hip_center)

        # 기울기 계산
        angle_shoulder_waist = compute_angle(shoulder_center, waist_center)
        angle_shoulder_hip = compute_angle(shoulder_center, hip_center)

        # feature 생성 (길이 8)
        feature = np.concatenate([
            shoulder_center,
            hip_center,
            waist_center,
            np.array([angle_shoulder_waist, angle_shoulder_hip])
        ])

        sequence.append(feature)

    cap.release()

    # --------------------------
    # None 값 보간 처리
    # --------------------------
    sequence = interpolate_missing(sequence)

    if sequence is None:
        print(f"⚠ {file}: 사람을 전혀 검출하지 못해 저장하지 않음")
        continue

    sequence = np.array(sequence)  # shape: (프레임수, 8)

    save_path = os.path.join(save_root, f"{file}_seq.npy")
    np.save(save_path, sequence)
    print(f"저장 완료 → {save_path}")
