import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import *


__all__ = ['wideresnet']

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

def wideresnet(depth=50, widen_factor=1, num_classes=10, norm=True, fixed=False, weight=None):
    return WideResNet(WideBasic, depth=depth, widen_factor=widen_factor, num_classes=num_classes, norm=norm, fixed=fixed, weight=weight)


if __name__ == '__main__':
    model = wideresnet(norm=True, num_classes=10)
    x = torch.randn(5, 3, 32, 32)
    print(model(x))