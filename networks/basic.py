import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod, ABCMeta

class Basic(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(Basic, self).__init__()

    @abstractmethod
    def get_body(self, x):
        """
        Get the input of the last fully-connected layer
        """
        pass

    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return

    def get_weight(self):
        """
        Get the weight of the last fully-connected layer
        """
        return self.linear.weight

    def get_margin(self, x):
        x = self.get_body(x)
        x = F.normalize(x, dim=1)
        weight = self.get_weight()
        weight = F.normalize(weight, dim=1)
        return F.linear(x, weight)

    def margin(self):
        weight = self.get_weight()
        norm = torch.sqrt(torch.sum(weight ** 2, dim=1))
        ratio = torch.max(norm) / torch.min(norm)

        tmp = F.normalize(weight, dim=1)
        similarity = torch.matmul(tmp, tmp.transpose(1, 0)) - 2 * torch.eye(tmp.size(0), device=weight.device)
        return torch.acos(torch.max(similarity)).item() / math.pi * 180, ratio.item()

    def forward(self, x):
        x = self.get_body(x)
        x = self.linear(x)
        return x

class NormLinear(nn.Module):
    """
        L2-normalization for both weights and inputs
    """
    __constants__ = ['bias', 'in_features', 'out_features']
    def __init__(self, in_features, out_features, bias=False):
        super(NormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight.data, a=math.sqrt(5))
        # torch.nn.init.normal_(self.weight)

    def forward(self, input):
        input = F.normalize(input, dim=1)
        weight = F.normalize(self.weight, dim=1)
        return F.linear(input, weight)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class FNormLinear(nn.Module):
    """
        L2-normalization for both weights and inputs when weights are fixed.
    """
    __constants__ = ['bias', 'in_features', 'out_features']
    def __init__(self, in_features, out_features, weight):
        super(FNormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.Tensor(out_features, in_features)
        self.set_parameters(weight)

    def set_parameters(self, weight):
        assert self.weight.size() == weight.size()
        self.weight = weight

    def forward(self, input):
        input = F.normalize(input, dim=1)
        weight = F.normalize(self.weight, dim=1)
        return F.linear(input, weight)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ConvBrunch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False, need_relu=True):
        super(ConvBrunch, self).__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
        ]
        if need_relu:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


