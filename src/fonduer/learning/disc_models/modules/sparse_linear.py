"""
A sparse linear module.
"""

import math

import torch
import torch.nn as nn


class SparseLinear(nn.Module):
    """A sparse linear layer.

    :param num_features: Size of features.
    :type num_features: int
    :param num_classes: Number of classes.
    :type num_classes: int
    :param bias: Use bias term or not.
    :type bias: bool
    :param padding_idx: padding index.
    :type padding_idx: int
    """

    def __init__(self, num_features, num_classes, bias=False, padding_idx=0):

        super(SparseLinear, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.padding_idx = padding_idx

        self.weight = nn.Embedding(
            self.num_features, self.num_classes, padding_idx=self.padding_idx
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_classes))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitiate the weight parameters.
        """

        stdv = 1.0 / math.sqrt(self.num_features)
        self.weight.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        if self.padding_idx is not None:
            self.weight.weight.data[self.padding_idx].fill_(0)

    def forward(self, x, w):
        """Forward function.

        :param x: Feature indices.
        :type x: torch.Tensor of shape (batch_size * length)
        :param w: Feature weights.
        :type w: torch.Tensor of shape (batch_size * length)
        :return: Output of linear layer.
        :rtype: torch.Tensor of shape (batch_size, num_classes)
        """

        if self.bias is None:
            return (w.unsqueeze(2) * self.weight(x)).sum(dim=1)
        else:
            return (w.unsqueeze(2) * self.weight(x)).sum(dim=1) + self.bias
