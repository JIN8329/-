"""
2단계: 시계열 데이터를 LSTM 입력형태 (윈도우)로 변환하는 코드
"""

import numpy as np
import os

sequence_root = r"C:\Users\HaJinYeong\2025-2\processed_sequences"
save_root = r"C:\Users\HaJinYeong\2025-2\lstm_input"
os.makedirs(save_root, exist_ok=True)

window = 30   # 30프레임 시퀀스
stride = 1

X_all = []
y_all = []

for file in os.listdir(sequence_root):
    if not file.endswith(".npy"):
        continue

    seq = np.load(os.path.join(sequence_root, file))  # (frames, 8)
    frames = len(seq)

    for i in range(0, frames - window, stride):
        X_window = seq[i:i+window]                  # (30, 8)
        X_all.append(X_window)

        # ----------------------
        # 단순 라벨링 기준 (예시)
        # 예: 기울기 급변하면 1 (낙상)
        # ----------------------
        angle1_start = X_window[0, 6]
        angle1_end   = X_window[-1, 6]

        angle_change = abs(angle1_end - angle1_start)

        y_all.append(1 if angle_change > 0.35 else 0)

X_all = np.array(X_all)
y_all = np.array(y_all)

np.save(os.path.join(save_root, "X.npy"), X_all)
np.save(os.path.join(save_root, "y.npy"), y_all)

print("변환 완료")
