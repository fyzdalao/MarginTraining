import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CosFaceLoss(nn.Module):
    def __init__(self, margin=0.35, scale=64):
        super(CosFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, logits, labels):
        index = torch.where(labels != -1)[0]
        one_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device)
        one_hot.scatter_(1, labels[index, None], 1)
        logits= logits - one_hot * self.margin
        logits = logits * self.scale

        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        return F.cross_entropy(logits - max_logits, labels)
