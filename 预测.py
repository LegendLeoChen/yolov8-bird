# —*-coding:utf-8 一*一
"""
作者:LegendLeo
日期:2024年03月27日
"""
from ultralytics import YOLO
from posehead.PoseHead import PoseHead
import torch, cv2

class_names = {0: 'stop', 1: 'fly'}
model = YOLO('runs/pose/train-对照组4/weights/best.pt')  # load a custom model
# model = YOLO('runs/pose/train-poseloss6/weights/best.pt')  # load a custom model

# Predict with the model
for i in range(6, 11):
    results = model(f'imgs/bird{i}.jpg')  # predict on an image
    for result in results:
        keypoints = result.keypoints  # Keypoints object for pose outputs

        posehead = PoseHead().to(torch.device('cuda'))
        posehead.load_state_dict(torch.load("posehead/posehead.pt"))
        posehead.eval()
        output = posehead(keypoints.xyn)
        for i in range(output.size(0)):             # 输出分类名
            predicted_class = output[i].argmax()
            class_name = class_names[predicted_class.item()]
            x1, y1, x2, y2 = result.boxes.xyxy[0][:4]  # 获取边界框的坐标信息
            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, class_name, (int(x1) + 20, int(y2) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)  # 在左上角写上类名
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            print(f"Sample {i + 1}: Predicted class is {class_name}")
            cv2.waitKey(0)

        # probs = result.probs  # Probs object for classification outputs
        # result.show()  # display to screen