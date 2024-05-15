# —*-coding:utf-8 一*一
"""
作者:LegendLeo
日期:2024年04月18日
"""
from ultralytics import YOLO
import onnx
# 加载训练好的模型
model = YOLO("runs/pose/train7/weights/best.pt")
# 将模型转为onnx格式
success = model.export(format='onnx')
model_file = 'runs/pose/train7/weights/best.onnx'
# 加载刚转换好的best.onnx文件
onnx_model = onnx.load(model_file)
# 重新保存为best2.onnx文件
onnx.save(onnx.shape_inference.infer_shapes(onnx_model), 'best2.onnx')