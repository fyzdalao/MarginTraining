import torch
import torch.nn as nn
import torch.nn.functional as F

class NormFaceLoss(nn.Module):
    def __init__(self, scale=64, weight=None):
        super(NormFaceLoss, self).__init__()
        self.scale = scale
        self.weight = weight

    def forward(self, logits, labels):
        logits = logits * self.scale
        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        return F.cross_entropy(logits - max_logits, labels, weight=self.weight)