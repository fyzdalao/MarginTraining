import torch

from .large_margin_softmax_loss import LargeMarginSoftmaxLoss

class SphereFaceLoss(LargeMarginSoftmaxLoss):
    def scale_logits(self, logits, embeddings, weight=None):
        embedding_norms = torch.norm(embeddings, dim=1)
        return logits * embedding_norms.unsqueeze(1) * self.scale