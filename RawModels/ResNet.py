# It is downloaded from https://github.com/junyuseu/pytorch-cifar-models/blob/master/models/resnet_cifar.py
#
# Reference:
# [1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
# [2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.


import math
import os
import sys
import torch
import torch.nn as nn

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from RawModels.basic_module import BasicModule

# Training parameters for CIFAR10
# global CIFAR10_Training_Parameters

CIFAR10_Training_Parameters = {
    'num_epochs': 200,
    'batch_size': 32,
    'lr': 1e-3
}


# adjust the learning rate for CIFAR10 training according to the number of epoch
def adjust_learning_rate(epoch, optimizer):
    minimum_learning_rate = 0.5e-6
    for param_group in optimizer.param_groups:
        lr_temp = param_group["lr"]
        if epoch == 80 or epoch == 120 or epoch == 160:
            lr_temp = lr_temp * 1e-1
        elif epoch == 180:
            lr_temp = lr_temp * 5e-1
        param_group["lr"] = max(lr_temp, minimum_learning_rate)
        print('The **learning rate** of the {} epoch is {}'.format(epoch, param_group["lr"]))


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(BasicModule):
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


class ResNet_Cifar(BasicModule):
    def __init__(self, block, layers, num_classes=10, thermometer=False, level=1):
        super(ResNet_Cifar, self).__init__()

        if thermometer is True:
            input_channels = 3 * level
        else:
            input_channels = 3

        self.inplanes = 16
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        # self.avgpool = nn.AvgPool2d(8, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

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
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion))

        layers = list([])
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

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = x - torch.max(x, dim=1, keepdim=True)[0]
        return x


def resnet20_cifar(thermometer=False, level=1):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], thermometer=thermometer, level=level)
    return model
