# —*-coding:utf-8 一*一
"""
作者:LegendLeo
日期:2024年04月07日
"""

import torch.nn as nn
import torch.nn.functional as F

class PoseHead(nn.Module):
    def __init__(self):
        super(PoseHead, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(16)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将输入形状从(batch_size, keypoints_num, 2)转换为(batch_size, 2, keypoints_num)
        x = self.norm(self.relu(self.conv1(x)))
        x = self.conv2(x)
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.relu(self.conv5(x))
        x = x.mean(dim=2)  # 对关键点维度求平均
        x = self.fc(x)
        return F.softmax(x, dim=1)