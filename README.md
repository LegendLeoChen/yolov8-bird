> 原项目：https://github.com/ultralytics/ultralytics

> 本项目数据集：

> 百度网盘：https://pan.baidu.com/s/1IAbgjSZs9G7XmKPEwMiBdw?pwd=1213 提取码：1213

> google：https://drive.google.com/drive/folders/12iIgp0_4aPTHtyqNYhfS4ZwWmwBjVXlE?usp=drive_link

# 概述
- 项目在YOLOv8基础上，以鸟类姿态检测为目标，设计了鸟类关键点，使用新的关键点回归损失函数，使用ViT模块改变网络结构，设计了外接的卷积输出网络将关键点映射到鸟类姿态分类，提供基于拼接的数据增强操作。
# 项目结构
- datasets：数据集及其工具的文件夹。
- imgs：预测用的图片、视频存放处。
- posehead：外接卷积输出网络的文件夹。
- runs：训练和验证结果存放处。
- ultralytics：YOLOv8本体。
- 其他文件夹不用详细了解。
# 重要文件说明
## 主文件夹
- 测试.ipynb：可训练（一般不用）、预测（图片或视频）、验证模型的脚本，视频预测针对的是添加了外接卷积输出网络的姿态分类任务。
- 人类.py：原项目对于人类的预测脚本，需要根据“改动”中进行更改。
- 对比效果.py：图片预测，同时用**两个模型**预测同样的图输出预测结果并排显示。
- 网络图生成.py：生成网络的详细图（onnx格式文件），包括张量形状、网络参数等信息，用于调试网络或概览网络。
- 训练.py：训练模型用的脚本。
- 预测.py：预测图片用的脚本。
- 验证.py：验证模型用的脚本。
## posehead文件夹
- PoseHead.py：外接卷积输出网络的网络结构。
- train.py：训练PoseHead用的脚本。
- posehead.pt：训练出来的模型。
## datasets文件夹
- background：存放背景图的文件夹，数据增强用到的背景。
- bird：训练YOLO网络用的数据集。
- bird_enhance：数据增强生成的图片及标注存放的文件夹。
- bird_pose：训练外接卷积输出网络PoseHead的数据集。
- 数据集转换.py：改变数据集格式，从train/valid/test下分别存放images/labels格式改为images/labels下分别是train/val/test格式。
- 数据增强.py：数据增强脚本，详见脚本内说明。
- 姿态分类标注.py：该脚本用于生成PoseHead网络数据集需要的标注文件，详见脚本内说明。
## ultralytics文件夹
- cfg/default.yaml：里面有模型训练、验证、预测各方面的配置、超参数等，可以在里面改也能在调用函数中设置。
- cfg/models/v8/yolov8-pose.yaml：存放YOLOv8姿态识别的网络结构，包括更改过的结构。
- cfg/datasets：里面存放各种数据集配置的yaml文件，包括本项目鸟类识别的bird.yaml文件。
- nn/modules：存放网络结构文件，其中clip_model.py是vit模型的结构是新加的。
- utils：loss.py有损失函数，metrics.py有评价指标。
# 改动操作
- 新的结构：ultralytics/cfg/models/v8/yolov8-pose.yaml，有多组backbone，使用一组注释其他组。
- 新的输出头：ultralytics/nn/modules/head.py，Pose类下self.cv4（161行左右），一组使用ViT，另一组没有。
- 新的关键点损失函数：ultralytics/utils/loss.py，KeypointLoss类下前向传播函数（138行），new_poseloss变量决定是否用新的损失函数。
- 鸟类和人类姿态识别更改：ultralytics/utils/plotting.py，Annotator类下kpts函数的bird变量，为True检测鸟类