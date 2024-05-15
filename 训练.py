import cv2, torch
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-pose.yaml')  # load a pretrained model (recommended for training)
model = YOLO('runs/pose/train-poseloss5/weights/best.pt')
# Train the model
results = model.train(data='bird.yaml', epochs=120, resume=False, lr0=0.008, warmup_epochs=0.0, patience=50, batch=16)