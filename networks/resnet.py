import torch
import torch.nn as nn
from .basic import  *
import torch.nn.functional as F

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # residual function
        self.residual_function = nn.Sequential(
            ConvBrunch(in_channels, out_channels, stride=stride),
            ConvBrunch(out_channels, out_channels * BasicBlock.expansion, need_relu=False)
        )
        # shortcut
        self.shortcut = nn.Sequential()
        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                ConvBrunch(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, need_relu=False)
            )

    def forward(self, x):
        return self.residual_function(x)+self.shortcut(x)

class BottleNeck(nn.Module):
    """
    Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.residual_function = nn.Sequential(
            ConvBrunch(in_channels, out_channels, kernel_size=1),
            ConvBrunch(out_channels, out_channels, stride=stride),
            ConvBrunch(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, need_relu=False)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                ConvBrunch(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, need_relu=False)
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)

class ResNet(Basic):
    def __init__(self, block, num_blocks, num_classes=10, mode='', weight=None):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.pre = ConvBrunch(3, 64)

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
            self.linear = nn.Linear(512 * block.expansion, num_classes, bias=False)
        self._reset_prams()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """
        # we have num_blocks blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
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

def resnet18(num_classes=10, mode='', weight=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, mode=mode, weight=weight)

def resnet34(num_classes=10, mode='', weight=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, mode=mode, weight=weight)


def resnet50(num_classes=10, mode='', weight=None):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, mode=mode, weight=weight)


def resnet101(num_classes=10, mode='', weight=None):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes,  mode=mode, weight=weight)


def resnet152(num_classes=10, mode='', weight=None):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes,  mode=mode, weight=weight)

if __name__ == '__main__':
    model = resnet18(norm=True, num_classes=10)
    x = torch.randn(5, 3, 32, 32)
    print(model(x))