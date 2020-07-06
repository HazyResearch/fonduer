"""Concat linear."""
from typing import Any, Dict, List

import torch
from torch import Tensor, nn as nn


class ConcatLinear(nn.Module):
    """Concat different outputs and feed into a linear layer.

    :param concat_output_keys: The keys of features to concat.
    :param input_dim: The total sum of input dim.
    :param outpt_dim: The output dim.
    """

    def __init__(
        self, concat_output_keys: List[str], input_dim: int, outpt_dim: int
    ) -> None:
        """Initialize ConcatLinear."""
        super().__init__()

        self.concat_output_keys = concat_output_keys
        self.linear = nn.Linear(input_dim, outpt_dim)

    def forward(self, intermediate_output_dict: Dict[str, Any]) -> Tensor:
        """Forward function."""
        input_feature = torch.cat(
            [intermediate_output_dict[key][0] for key in self.concat_output_keys], dim=1
        )
        return self.linear(input_feature)
