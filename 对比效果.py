# —*-coding:utf-8 一*一
"""
作者:LegendLeo
日期:2024年05月04日
"""
from ultralytics import YOLO
from posehead.PoseHead import PoseHead
import torch, cv2

class_names = {0: 'stop', 1: 'fly'}
# Load both YOLO models
model1 = YOLO('runs/pose/train-对照组4/weights/best.pt')
model2 = YOLO('runs/pose/train-综合/weights/best.pt')

# Start the loop
for i in range(2, 11):
    # Load the image
    img = cv2.imread(f'imgs/bird{i}.jpg')

    # Predict with the first model
    results1 = model1(img)
    for result in results1:
        keypoints = result.keypoints  # Keypoints object for pose outputs

        posehead = PoseHead().to(torch.device('cuda'))
        posehead.load_state_dict(torch.load("posehead/posehead.pt"))
        posehead.eval()
        output = posehead(keypoints.xyn)
        for i in range(output.size(0)):  # 输出分类名
            predicted_class = output[i].argmax()
            class_name = class_names[predicted_class.item()]
            x1, y1, x2, y2 = result.boxes.xyxy[0][:4]  # 获取边界框的坐标信息
            annotated_frame1 = result[0].plot()
            cv2.putText(annotated_frame1, class_name, (int(x1) + 20, int(y2) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)  # 在左上角写上类名

    # Predict with the second model
    results2 = model2(img)
    for result in results2:
        keypoints = result.keypoints  # Keypoints object for pose outputs

        posehead = PoseHead().to(torch.device('cuda'))
        posehead.load_state_dict(torch.load("posehead/posehead.pt"))
        posehead.eval()
        output = posehead(keypoints.xyn)
        for i in range(output.size(0)):  # 输出分类名
            predicted_class = output[i].argmax()
            class_name = class_names[predicted_class.item()]
            x1, y1, x2, y2 = result.boxes.xyxy[0][:4]  # 获取边界框的坐标信息
            annotated_frame2 = result[0].plot()
            cv2.putText(annotated_frame2, class_name, (int(x1) + 20, int(y2) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)  # 在左上角写上类名

    # Create a new image with both models' predictions side by side
    combined_img = cv2.hconcat([annotated_frame1, annotated_frame2])

    # Display the combined image
    cv2.imshow("YOLOv8 Inference - Model 1 vs Model 2", combined_img)
    cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()