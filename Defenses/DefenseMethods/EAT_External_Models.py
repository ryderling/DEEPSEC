#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/9/19 18:26
# @Author  : Jiaxu Zou & Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : EAT_External_Models.py
# **************************************

'''
The cifar10 model frameworks are downloaded from
https://github.com/junyuseu/pytorch-cifar-models/blob/master/models/resnet_cifar.py
and
https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
[3] H. Gao, Z. Liu, L. Maaten and K. Weinberger. Densely connected convolutional networks. In CVPR, 2017.
[4] Y. Lecun, L. Bottou, Y. Bengio and P. Haffner. Gradient-Based Learning Applied to Document Recognition. In IEEE, 1998.
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from RawModels.basic_module import BasicModule


class MNIST_A(BasicModule):
    def __init__(self):
        super(MNIST_A, self).__init__()

        self.conv32 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
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

    def forward(self, x):
        out = self.conv32(x)
        out = self.conv64(out)
        out = out.view(-1, 4 * 4 * 64)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class MNIST_B(BasicModule):
    def __init__(self):
        super(MNIST_B, self).__init__()

        self.conv64 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=8, stride=(2, 2), padding=(17, 17)),
            nn.ReLU()
        )
        self.conv128_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=(2, 2)),
            nn.ReLU()
        )
        self.conv128_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5),
            nn.ReLU()
        )
        self.fc = nn.Linear(8 * 8 * 128, 10)
        self.dropout_1 = nn.Dropout2d(p=0.2)
        self.dropout_2 = nn.Dropout2d(p=0.5)

    def forward(self, x):
        out = self.dropout_1(x)
        out = self.conv64(out)
        out = self.conv128_1(out)
        out = self.conv128_2(out)
        out = self.dropout_2(out)
        out = out.view(-1, 8 * 8 * 128)
        out = self.fc(out)
        return out


class MNIST_C(BasicModule):
    def __init__(self):
        super(MNIST_C, self).__init__()

        self.conv128 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3),
            nn.ReLU()
        )
        self.conv64 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(24 * 24 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout_1 = nn.Dropout2d(p=0.25)
        self.dropout_2 = nn.Dropout2d(p=0.5)

    def forward(self, x):
        out = self.conv128(x)
        out = self.conv64(out)
        out = self.dropout_1(out)
        out = out.view(-1, 24 * 24 * 64)
        out = F.relu(self.fc1(out))
        out = self.dropout_2(out)
        out = self.fc2(out)
        return out


class MNIST_D(BasicModule):
    def __init__(self):
        super(MNIST_D, self).__init__()

        self.fc1 = nn.Linear(28 * 28 * 1, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 10)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        out = x.view(-1, 28 * 28 * 1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out


# below is the model framework for cifar10

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CIFAR10_A(BasicModule):
    def __init__(self):
        super(CIFAR10_A, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 16, 3)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * BasicBlock.expansion, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# DenseNet

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class CIFAR10_B(BasicModule):
    # DenseNet121
    def __init__(self, block=Bottleneck, nblocks=[6, 12, 24, 16], growth_rate=32, reduction=0.5, num_classes=10):
        super(CIFAR10_B, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class CIFAR10_C(BasicModule):
    def __init__(self):
        super(CIFAR10_C, self).__init__()

        self.conv64 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv128 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(5 * 5 * 128, 256)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.conv64(x)
        out = self.conv128(out)
        out = out.view(-1, 5 * 5 * 128)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

#LeNet

class CIFAR10_D(BasicModule):
    def __init__(self):
        super(CIFAR10_D, self).__init__()

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        out = self.conv6(x)
        out = self.conv16(out)
        out = out.view(-1, 5 * 5 * 16)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # out = self.dropout(out)
        out = self.fc3(out)
        return out
