from typing import Any

from pytorch_toolbelt.losses import BinaryFocalLoss
from torch import nn
from torch.nn.modules.loss import BCEWithLogitsLoss


class WeightedLosses(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, *input: Any, **kwargs: Any):
        cum_loss = 0
        for loss, w in zip(self.losses, self.weights):
            cum_loss += w * loss.forward(*input, **kwargs)
        return cum_loss


class BinaryCrossentropy(BCEWithLogitsLoss):
    pass


class FocalLoss(BinaryFocalLoss):
    def __init__(self, alpha=None, gamma=3, ignore_index=None, reduction="mean", normalized=False,
                 reduced_threshold=None):
        super().__init__(alpha, gamma, ignore_index, reduction, normalized, reduced_threshold)