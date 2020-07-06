"""Soft cross entropy loss."""
from typing import List

import torch
from torch import Tensor, nn as nn
from torch.nn import functional as F


class SoftCrossEntropyLoss(nn.Module):
    """Calculate the CrossEntropyLoss with soft targets.

    :param weight: Weight to assign to each of the classes. Default: None
    :param reduction: The way to reduce the losses: 'none' | 'mean' | 'sum'.
        'none': no reduction,
        'mean': the mean of the losses,
        'sum': the sum of the losses.
    """

    def __init__(self, weight: List[float] = None, reduction: str = "mean"):
        """Initialize SoftCrossEntropyLoss."""
        super().__init__()
        if weight is None:
            self.weight = None
        else:
            self.register_buffer("weight", torch.tensor(weight))

        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Calculate the loss.

        :param input: prediction logits
        :param target: target probabilities
        :return: loss
        """
        n, k = input.shape
        losses = input.new_zeros(n)

        for i in range(k):
            cls_idx = input.new_full((n,), i, dtype=torch.long)
            loss = F.cross_entropy(input, cls_idx, reduction="none")
            if self.weight is not None:
                loss = loss * self.weight[i]
            losses += target[:, i].float() * loss

        if self.reduction == "mean":
            losses = losses.mean()
        elif self.reduction == "sum":
            losses = losses.sum()
        elif self.reduction != "none":
            raise ValueError(f"Unrecognized reduction: {self.reduction}")

        return losses
