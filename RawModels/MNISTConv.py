#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/9/7 22:04
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : MNISTConv.py 
# **************************************

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from RawModels.basic_module import BasicModule

# Training parameters for MNIST
MNIST_Training_Parameters = {
    'num_epochs': 20,
    'batch_size': 100,
    'learning_rate': 0.05,
    'momentum': 0.9,
    'decay': 1e-6
}


# define the network architecture for MNIST
class MNISTConvNet(BasicModule):
    def __init__(self, thermometer=False, level=1):
        super(MNISTConvNet, self).__init__()

        if thermometer is True:
            input_channels = 1 * level
        else:
            input_channels = 1

        self.conv32 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv64 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(4 * 4 * 64, 200)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        # softmax ? or not

    def forward(self, x):
        out = self.conv32(x)
        out = self.conv64(out)
        out = out.view(-1, 4 * 4 * 64)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = out - torch.max(out, dim=1, keepdim=True)[0]
        return out
