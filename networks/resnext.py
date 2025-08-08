
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import *


CARDINALITY = 32
DEPTH = 4
BASEWIDTH = 64


__all__ = ['resnxet18', 'resnxet34', 'resnxet50', 'resnxet101', 'resnet152']

class ResNextBasic(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNextBasic, self).__init__()
        # residual function
        C = CARDINALITY
        D = int(DEPTH * out_channels / BASEWIDTH)
        self.split_transforms = nn.Sequential(
            ConvBrunch(in_channels, C*D, groups=C),
            ConvBrunch(C*D, out_channels * ResNextBasic.expansion, need_relu=False)
        )
        # shortcut
        self.shortcut = nn.Sequential()
        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != ResNextBasic.expansion * out_channels:
            self.shortcut = nn.Sequential(
                ConvBrunch(in_channels, out_channels * ResNextBasic.expansion, kernel_size=1, need_relu=False)
            )

    def forward(self, x):
        return F.relu(self.split_transforms(x) + self.shortcut(x), inplace=True)



class ResNextBottoleNeckC(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride):
        super(ResNextBottoleNeckC, self).__init__()
        C = CARDINALITY
        D = int(DEPTH * out_channels / BASEWIDTH)
        self.split_transforms = nn.Sequential(
            ConvBrunch(in_channels, C*D, kernel_size=1, groups=C),
            ConvBrunch(C * D, C * D, kernel_size=3, stride=stride, groups=C),
            ConvBrunch(C * D, out_channels, kernel_size=1, need_relu=False)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * ResNextBottoleNeckC.expansion:
            self.shortcut = ConvBrunch(in_channels, out_channels * ResNextBottoleNeckC.expansion, stride=stride, kernel_size=1, need_relu=False)

    def forward(self, x):
        return F.relu(self.split_transforms(x) + self.shortcut(x), inplace=True)

class ResNext(Basic):
    def __init__(self, block, num_blocks, num_classes=10, mode='', weight=None):
        super(ResNext, self).__init__()
        self.in_channels = 64
        self.pre = ConvBrunch(3, 64, kernel_size=3)

        self.stage1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.stage2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.stage3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.stage4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if mode == 'norm':
            self.linear = NormLinear(512 * block.expansion, num_classes)
        elif mode == 'fix':
            self.linear = FNormLinear(512 * block.expansion, num_classes, weight)
        else:
            self.linear = nn.Linear(512 * block.expansion, num_classes)
        self._reset_prams()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def get_body(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

def resnxet18(num_classes=10, mode='', weight=None):
    return ResNext(ResNextBasic, [2, 2, 2, 2], num_classes=num_classes, mode=mode, weight=weight)

def resnxet34(num_classes=10, mode='', weight=None):
    return ResNext(ResNextBasic, [3, 4, 6, 3], num_classes=num_classes, mode=mode, weight=weight)


def resnxet50(num_classes=10, mode='', weight=None):
    return ResNext(ResNextBottoleNeckC, [3, 4, 6, 3], num_classes=num_classes, mode=mode, weight=weight)


def resnxet101(num_classes=10, mode='', weight=None):
    return ResNext(ResNextBottoleNeckC, [3, 4, 23, 3], num_classes=num_classes, mode=mode, weight=weight)


def resnet152(num_classes=10, mode='', weight=None):
    return ResNext(ResNextBottoleNeckC, [3, 8, 36, 3], num_classes=num_classes, mode=mode, weight=weight)


if __name__ == '__main__':
    model = resnxet18(mode='', num_classes=10)
    x = torch.randn(5, 3, 32, 32)
    print(model.get_margin(x))