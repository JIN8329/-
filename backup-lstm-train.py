"""
3단계: LSTM 모델 학습 코드 (필수 수정 완료)
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
# ① 데이터 섞기 반드시 필요
# --------------------------
X, y = shuffle(X, y, random_state=42)

# --------------------------
# ② 클래스 불균형 처리
# --------------------------
normal_count = np.sum(y == 0)
fall_count   = np.sum(y == 1)

class_weight = {
    0: 1.0,
    1: normal_count / fall_count   # fall이 적으므로 더 높은 가중
}

print("class_weight:", class_weight)

# --------------------------
# LSTM 모델 구성
# --------------------------
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(30, 8)),
    Dropout(0.3),                     # ★ 추가
    Dense(32, activation='relu'),
    Dropout(0.3),                     # ★ 추가
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --------------------------
# Early Stopping
# --------------------------
es = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')

# --------------------------
# 학습
# --------------------------
model.fit(
    X, y,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es],
    class_weight=class_weight      # ★ 추가됨
)

model.save("fall_lstm_model.h5")
print("모델 저장 완료")
