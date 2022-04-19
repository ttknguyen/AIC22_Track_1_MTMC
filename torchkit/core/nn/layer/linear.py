#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Linear layer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch import Tensor

from torchkit.core.factory import LINEAR_LAYERS

__all__ = [
    "Linear"
]


# MARK: - Linear

@LINEAR_LAYERS.register(name="linear")
class Linear(nn.Linear):
    """Applies a linear transformation to the incoming data: `y = xA^T + b`

    Wraps torch.nn.Linear to support AMP + torchscript usage by manually
    casting weight & bias to input.dtype to work around an issue w/
    torch.addmm in this use case.
    """

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)


# MARK: - Builder

LINEAR_LAYERS.register(name="identity",    module=nn.Identity)
LINEAR_LAYERS.register(name="bilinear",    module=nn.Bilinear)
LINEAR_LAYERS.register(name="lazy_linear", module=nn.LazyLinear)
