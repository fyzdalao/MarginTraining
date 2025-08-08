import torch
import torch.nn as nn
import torch.nn.functional as F

class SampleMarginLoss(nn.Module):
    def __init__(self):
        super(SampleMarginLoss, self).__init__()

    def forward(self, logits, labels):
        label_one_hot = F.one_hot(labels, logits.size()[1]).float().to(logits.device)
        l1 = torch.sum(logits * label_one_hot, dim=1)
        tmp = logits * (1-label_one_hot) - label_one_hot
        l2 = torch.max(tmp, dim=1)[0]
        loss = l2 - l1
        return loss.mean()
