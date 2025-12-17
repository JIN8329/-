"""
3단계: 개선된 LSTM 모델 학습 코드 (최종 추천)
"""

import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------
# 데이터 로드
# --------------------------
X = np.load("C:/Users/HaJinYeong/2025-2/lstm_input/X.npy")   # (N, 30, 8)
y = np.load("C:/Users/HaJinYeong/2025-2/lstm_input/y.npy")

# --------------------------
# ① 입력 데이터 정규화 (매우 중요)
# --------------------------
X_mean = X.mean()
X_std = X.std() + 1e-6        # 0 나누기 방지
X = (X - X_mean) / X_std

# --------------------------
# ② 데이터 셔플
# --------------------------
X, y = shuffle(X, y, random_state=42)

# --------------------------
# ③ 클래스 불균형 처리 (개선 공식)
# --------------------------
normal_count = np.sum(y == 0)
fall_count   = np.sum(y == 1)
total = len(y)

class_weight = {
    0: total / (2 * normal_count),
    1: total / (2 * fall_count)
}

print("class_weight:", class_weight)

# --------------------------
# ④ LSTM 모델 (2단 LSTM 구조)
# --------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 8)),
    Dropout(0.3),

    LSTM(32, return_sequences=False),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --------------------------
# ⑤ Early Stopping
# --------------------------
es = EarlyStopping(
    patience=10,
    restore_best_weights=True,
    monitor='val_loss'
)

# --------------------------
# ⑥ 모델 학습 (batch ↓)
# --------------------------
model.fit(
    X, y,
    epochs=40,
    batch_size=16,              # 배치 줄임 (성능 향상)
    validation_split=0.2,
    callbacks=[es],
    class_weight=class_weight
)

model.save("fall_lstm_model.h5")
print("모델 저장 완료")
