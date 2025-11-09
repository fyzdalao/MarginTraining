import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import *


__all__ = [
    'wideresnet',
    'wideresnet22_10',
    'wideresnet28',
    'wideresnet28_8',
    'wideresnet40_6',
    'wideresnet40_7',
    'wideresnet44_6',
    'wideresnet46_6',
    'wideresnet50',
    'wideresnet50_5',
    'wideresnet52_5',
    'wideresnet54_5',
    'wideresnet56_5',
    'wideresnet58_5',
    'wideresnet70_4',
    'wideresnet72_4',
    'wideresnet76_4',
    'wideresnet80_4',
    'wideresnet82_4'
]

class WideBasic(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(WideBasic, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            ConvBrunch(in_channels, out_channels, stride=stride, bias=True),
            nn.Dropout(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class WideResNet(Basic):
    def __init__(self, block, depth=50, widen_factor=1, num_classes=10, mode='', weight=None):
        super(WideResNet, self).__init__()
        self.depth = depth
        k = widen_factor
        l = int((depth-4) / 6)
        self.in_channels = 16
        self.pre = nn.Conv2d(3, self.in_channels, 3, 1, padding=1)
        self.stage1 = self._make_layer(block, 16 * k, l, 1)
        self.stage2 = self._make_layer(block, 32 * k, l, 2)
        self.stage3 = self._make_layer(block, 64 * k, l, 2)
        self.stage4 = nn.Sequential(
            nn.BatchNorm2d(64 * k),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if mode == 'norm':
            self.linear = NormLinear(64 * k, num_classes)
        elif mode == 'fix':
            self.linear = FNormLinear(64 * k, num_classes, weight)
        else:
            self.linear = nn.Linear(64 * k, num_classes)
        self._reset_prams()

    def get_body(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

def wideresnet(num_classes=10, mode='', weight=None, depth=28, widen_factor=10):
    return WideResNet(WideBasic, depth=depth, widen_factor=widen_factor, num_classes=num_classes, mode=mode, weight=weight)


def wideresnet28(num_classes=10, mode='', weight=None, widen_factor=10):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=28, widen_factor=widen_factor)


def wideresnet50(num_classes=10, mode='', weight=None, widen_factor=10):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=50, widen_factor=widen_factor)


def wideresnet22_10(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=22, widen_factor=10)


def wideresnet28_8(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=28, widen_factor=8)


def wideresnet40_6(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=40, widen_factor=6)

def wideresnet40_7(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=40, widen_factor=7)

def wideresnet44_6(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=44, widen_factor=6)

def wideresnet46_6(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=46, widen_factor=6)


def wideresnet50_5(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=50, widen_factor=5)

def wideresnet52_5(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=52, widen_factor=5)

def wideresnet54_5(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=54, widen_factor=5)

def wideresnet56_5(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=56, widen_factor=5)

def wideresnet58_5(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=58, widen_factor=5)


def wideresnet70_4(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=70, widen_factor=4)

def wideresnet72_4(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=72, widen_factor=4)


def wideresnet76_4(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=76, widen_factor=4)

def wideresnet80_4(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=80, widen_factor=4)

def wideresnet82_4(num_classes=10, mode='', weight=None):
    return wideresnet(num_classes=num_classes, mode=mode, weight=weight, depth=82, widen_factor=4)


if __name__ == '__main__':
    model = wideresnet(norm=True, num_classes=10)
    x = torch.randn(5, 3, 32, 32)
    print(model(x))