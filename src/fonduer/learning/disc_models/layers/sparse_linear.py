"""
A sparse linear module.
"""

import math

import torch
import torch.nn as nn


class SparseLinear(nn.Module):
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
        stdv = 1. / math.sqrt(self.num_features)
        self.weight.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        if self.padding_idx is not None:
            self.weight.weight.data[self.padding_idx].fill_(0)

    def forward(self, x, w):
        """
        x : batch_size * length, the feature indices
        w : batch_size * length, the weight for each feature
        """
        if self.bias is None:
            return (w.unsqueeze(2) * self.weight(x)).sum(dim=1)
        else:
            return (w.unsqueeze(2) * self.weight(x)).sum(dim=1) + self.bias
