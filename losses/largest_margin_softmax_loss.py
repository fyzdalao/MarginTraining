import torch
import torch.nn as nn
import torch.nn.functional as F

class LMSoftmaxLoss(nn.Module):
    def __init__(self, scale=64, weight=None):
        super(LMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.weight = weight
        self.eps = 1e-7

    def forward(self, logits, labels):
        mask = F.one_hot(labels, logits.size()[1]).float().to(logits.device)
        # print(logits)
        logits = logits * self.scale
        l1 = torch.sum(logits * mask, dim=1, keepdim=True)
        diff = logits - l1
        logits_exp = torch.exp(diff)
        l2 = torch.sum(logits_exp * (1-mask), dim=1)
        loss = torch.log(l2)
        if self.weight is not None:
            weight = self.weight.gather(0, labels.view(-1))
            loss = loss * weight
        return loss.mean() / self.scale

if __name__ == '__main__':
    # torch.manual_seed(123)
    criterion = LNormFaceLoss()
    x = torch.randint(-100, 100, (5, 10))/100
    y = torch.LongTensor([0, 0, 1, 2, 1])
    print(criterion(x, y))