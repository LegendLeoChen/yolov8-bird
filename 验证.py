# —*-coding:utf-8 一*一
"""
作者:LegendLeo
日期:2024年04月10日
"""
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-pose.pt')  # load an official model
model = YOLO('runs/pose/train-综合/weights/best.pt')  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
print(metrics.pose.map)    # map50-95
print(metrics.pose.map50)  # map50
print(metrics.pose.map75)  # map75