from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch import Tensor


class Sum_module(nn.Module):
    def __init__(self, sum_output_keys: List[str]) -> None:
        super().__init__()

        self.sum_output_keys = sum_output_keys

    def forward(  # type:ignore
        self, intermediate_output_dict: Dict[str, Any]
    ) -> Tensor:
        return torch.stack(
            [intermediate_output_dict[key][0] for key in self.sum_output_keys], dim=0
        ).sum(dim=0)
