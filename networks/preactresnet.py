import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import *

__all__ = ['preactresnet18', 'preactresnet34', 'preactresnet50', 'preactresnet101', 'preactresnet152']


class PreActBasic(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride):
        super(PreActBasic, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            ConvBrunch(in_channels, out_channels, stride=stride, bias=True),
            nn.Conv2d(out_channels, out_channels * PreActBasic.expansion, kernel_size=3, padding=1)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBasic.expansion, 1, stride=stride)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class PreActBottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride):
        super(PreActBottleNeck, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            ConvBrunch(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
            ConvBrunch(out_channels, out_channels, bias=True),
            nn.Conv2d(out_channels, out_channels * PreActBottleNeck.expansion, 1)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBottleNeck.expansion, 1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        s = self.shortcut(x)
        return res + s


class PreActResNet(Basic):
    def __init__(self, block, num_blocks, num_classes=10, mode='', weight=None):
        super(PreActResNet, self).__init__()
        self.in_channels = 64
        self.pre = ConvBrunch(3, 64, kernel_size=3, bias=True)

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

def preactresnet18(num_classes=10, mode='', weight=None):
    return PreActResNet(PreActBasic, [2, 2, 2, 2], num_classes=num_classes, mode=mode, weight=weight)

def preactresnet34(num_classes=10, mode='', weight=None):
    return PreActResNet(PreActBasic, [3, 4, 6, 3], num_classes=num_classes, mode=mode, weight=weight)


def preactresnet50(num_classes=10, mode='', weight=None):
    return PreActResNet(PreActBottleNeck, [3, 4, 6, 3], num_classes=num_classes, mode=mode, weight=weight)


def preactresnet101(num_classes=10, mode='', weight=None):
    return PreActResNet(PreActBottleNeck, [3, 4, 23, 3], num_classes=num_classes, mode=mode, weight=weight)


def preactresnet152(num_classes=10, mode='', weight=None):
    return PreActResNet(PreActBottleNeck, [3, 8, 36, 3], num_classes=num_classes, mode=mode, weight=weight)


if __name__ == '__main__':
    model = preactresnet18(mode='norm', num_classes=10)
    x = torch.randn(5, 3, 32, 32)
    print(model.get_margin(x))