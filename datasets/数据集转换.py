"""
作者:LegendLeo
日期:2024年03月27日

功能说明：
改变数据集格式，从train/valid/test下分别存放images/labels格式改为images/labels下分别是train/val/test格式。
"""
import os
import shutil

# 定义原始数据集文件夹路径和目标数据集文件夹路径
original_data_dir = 'bird2'
target_data_dir = 'bird'

# 创建目标数据集文件夹和子文件夹
os.makedirs(os.path.join(target_data_dir, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(target_data_dir, 'labels', 'test'), exist_ok=True)
os.makedirs(os.path.join(target_data_dir, 'labels', 'val'), exist_ok=True)
os.makedirs(os.path.join(target_data_dir, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(target_data_dir, 'images', 'test'), exist_ok=True)
os.makedirs(os.path.join(target_data_dir, 'images', 'val'), exist_ok=True)

# 复制train数据
for filename in os.listdir(os.path.join(original_data_dir, 'train', 'labels')):
    shutil.copy(os.path.join(original_data_dir, 'train', 'labels', filename),
                os.path.join(target_data_dir, 'labels', 'train', filename))
for filename in os.listdir(os.path.join(original_data_dir, 'train', 'images')):
    shutil.copy(os.path.join(original_data_dir, 'train', 'images', filename),
                os.path.join(target_data_dir, 'images', 'train', filename))

# 复制test数据
for filename in os.listdir(os.path.join(original_data_dir, 'test', 'labels')):
    shutil.copy(os.path.join(original_data_dir, 'test', 'labels', filename),
                os.path.join(target_data_dir, 'labels', 'test', filename))
for filename in os.listdir(os.path.join(original_data_dir, 'test', 'images')):
    shutil.copy(os.path.join(original_data_dir, 'test', 'images', filename),
                os.path.join(target_data_dir, 'images', 'test', filename))

# 复制val数据
for filename in os.listdir(os.path.join(original_data_dir, 'valid', 'labels')):
    shutil.copy(os.path.join(original_data_dir, 'valid', 'labels', filename),
                os.path.join(target_data_dir, 'labels', 'val', filename))
for filename in os.listdir(os.path.join(original_data_dir, 'valid', 'images')):
    shutil.copy(os.path.join(original_data_dir, 'valid', 'images', filename),
                os.path.join(target_data_dir, 'images', 'val', filename))