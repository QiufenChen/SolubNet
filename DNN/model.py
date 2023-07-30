# Author QFIUNE
# coding=utf-8
# @Time: 2022/8/31 20:41
# @File: model.py
# @Software: PyCharm
# @contact: 1760812842@qq.com

import torch

import torch.nn as nn


class LinearNet(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(LinearNet, self).__init__()

        self.model = torch.nn.Sequential(
            nn.Linear(n_feature, 512),
            nn.Sigmoid(),
            # nn.Dropout(0.5),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.Sigmoid(),
            # nn.Dropout(0.5),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 2048),
            nn.Sigmoid(),
            # nn.Dropout(0.5),
            nn.BatchNorm1d(2048),

            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.BatchNorm1d(4096),

            nn.Linear(4096, n_output))


    def forward(self, x):
        x = self.model(x)
        return x
