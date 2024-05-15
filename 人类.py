# —*-coding:utf-8 一*一
"""
作者:LegendLeo
日期:2024年04月29日
"""
from ultralytics import YOLO
import cv2
from posehead.PoseHead import PoseHead
import torch
model = YOLO('yolov8n-pose.pt')

video_path = 0
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        results = model(frame)
        # Visualize the results on the frame
        for i, result in enumerate(results):
            if i != 0:
                annotated_frame = annotated_frame.plot()
            else:
                annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()