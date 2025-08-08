import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import *

__all__ = ['cnn']

class CNN(Basic):
    def __init__(self, show=False, mode='', weight=None, use_relu=True):
        super(CNN, self).__init__()
        self.type = type
        if show:
            embed_dim = 3
        else:
            embed_dim = 128
        self.block = nn.Sequential(
            ConvBrunch(1, 32, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBrunch(32, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        if use_relu:
            self.fc1 = nn.Sequential(
                nn.Linear(64 * 7 * 7, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(64 * 7 * 7, embed_dim),
                nn.BatchNorm1d(embed_dim),
            )
        if mode == 'norm':
            self.linear = NormLinear(embed_dim, 10, bias=False)
        elif mode == 'fix':
            self.linear = FNormLinear(embed_dim, 10, weight)
        else:
            self.linear = nn.Linear(embed_dim, 10, bias=False)
        self.fc_size = 64 * 7 * 7
        self._reset_prams()

    def get_body(self, x):
        x = self.block(x)
        x = x.view(-1, self.fc_size)
        x = self.fc1(x)
        return x

    def get_weight(self):
        return self.linear.weight

def cnn(show=False, mode='', weight=None, use_relu=True):
    return CNN(show=show, mode=mode, weight=weight, use_relu=use_relu)