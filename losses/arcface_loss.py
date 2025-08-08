import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, margin=0.1, scale=64, weight=None):
        super(ArcFaceLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.eps = 1e-7

    def forward(self, cosine, labels):
        mask = F.one_hot(labels, cosine.size()[1]).float().to(cosine.device)
        cosine_of_target_classes = cosine[mask == 1]
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1 + self.eps, 1 - self.eps))
        modified_cosine_of_target_classes = torch.cos(angles + self.margin)
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)
        logits = cosine + (mask * diff)
        logits = logits * self.scale
        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        return F.cross_entropy(logits - max_logits, labels)

# class ArcFaceLoss(nn.Module):
#     def __init__(self, margin=0.5, scale=64):
#         super(ArcFaceLoss, self).__init__()
#         self.scale = scale
#         self.margin = margin
#
#     def forward(self, cosine: torch.Tensor, label):
#         index = torch.where(label != -1)[0]
#         m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
#         m_hot.scatter_(1, label[index, None], self.margin)
#         cosine.acos_()
#         cosine[index] += m_hot
#
#         cosine.cos_().mul_(self.scale)
#         max_cosine = torch.max(cosine, dim=1, keepdim=True)[0]
#         return F.cross_entropy(cosine-max_cosine, label)

