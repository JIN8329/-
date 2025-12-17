import torch
from ultralytics import YOLO

# PyTorch GPU 확인
print("PyTorch CUDA 사용 가능:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 이름:", torch.cuda.get_device_name(0))
    print("CUDA 버전:", torch.version.cuda)

# YOLO GPU 확인
model = YOLO("yolov8n-pose.pt")
print("YOLO device:", model.device)  # cuda:0 가 나와야 GPU 사용
