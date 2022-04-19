#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Selectable adaptive pooling with the ability to select the type of
pooling from:
- `avg`     - Average pooling.
- `max`     - Max pooling.
- `avgmax`  - Sum of average and max pooling re-scaled by 0.5.
- `avgmaxc` - Concatenation of average and max pooling along feature dim,
doubles feature dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchkit.core.factory import POOL_LAYERS

__all__ = [
    "adaptive_avgmax_pool2d",
    "adaptive_catavgmax_pool2d",
    "adaptive_pool_feat_mult",
    "AdaptiveAvgMaxPool",
    "AdaptiveAvgMaxPool2d",
    "AdaptiveCatAvgMaxPool",
    "AdaptiveCatAvgMaxPool2d",
    "FastAdaptiveAvgPool",
    "FastAdaptiveAvgPool2d",
    "select_adaptive_pool2d", 
    "SelectAdaptivePool",
    "SelectAdaptivePool2d"
]


# MARK: - FastAdaptiveAvgPool2d

@POOL_LAYERS.register(name="fast_adaptive_avg_pool2d")
class FastAdaptiveAvgPool2d(nn.Module):

    # MARK: Magic Functions

    def __init__(self, flatten: bool = False):
        super().__init__()
        self.flatten = flatten

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        return input.mean((2, 3), keepdim=not self.flatten)


FastAdaptiveAvgPool = FastAdaptiveAvgPool2d
POOL_LAYERS.register(name="fast_adaptive_avg_pool", module=FastAdaptiveAvgPool)


# MARK: - AdaptiveAvgMaxPool2d

def adaptive_avgmax_pool2d(input: Tensor, output_size: int = 1):
    x_avg = F.adaptive_avg_pool2d(input, output_size)
    x_max = F.adaptive_max_pool2d(input, output_size)
    return 0.5 * (x_avg + x_max)


@POOL_LAYERS.register(name="adaptive_avg_max_pool2d")
class AdaptiveAvgMaxPool2d(nn.Module):

    # MARK: Magic Functions

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        return adaptive_avgmax_pool2d(input, self.output_size)


AdaptiveAvgMaxPool = AdaptiveAvgMaxPool2d
POOL_LAYERS.register(name="adaptive_avg_max_pool", module=AdaptiveAvgMaxPool)


# MARK: - AdaptiveCatAvgMaxPool2d

def adaptive_catavgmax_pool2d(input: Tensor, output_size: int = 1):
    x_avg = F.adaptive_avg_pool2d(input, output_size)
    x_max = F.adaptive_max_pool2d(input, output_size)
    return torch.cat((x_avg, x_max), 1)


@POOL_LAYERS.register(name="adaptive_cat_avg_max_pool2d")
class AdaptiveCatAvgMaxPool2d(nn.Module):

    # MARK: Magic Functions

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        return adaptive_catavgmax_pool2d(input, self.output_size)


AdaptiveCatAvgMaxPool = AdaptiveCatAvgMaxPool2d
POOL_LAYERS.register(name="adaptive_cat_avg_max_pool", module=AdaptiveCatAvgMaxPool)


# MARK: - SelectAdaptivePool2d

def adaptive_pool_feat_mult(pool_type: str = "avg") -> int:
    if pool_type == "catavgmax":
        return 2
    else:
        return 1
    
    
def select_adaptive_pool2d(
    input: Tensor, pool_type: str = "avg", output_size: int = 1
):
    """Selectable global pooling function with dynamic input kernel size."""
    if pool_type == "avg":
        input = F.adaptive_avg_pool2d(input, output_size)
    elif pool_type == "avgmax":
        input = adaptive_avgmax_pool2d(input, output_size)
    elif pool_type == "catavgmax":
        input = adaptive_catavgmax_pool2d(input, output_size)
    elif pool_type == "max":
        input = F.adaptive_max_pool2d(input, output_size)
    elif True:
        raise ValueError("Invalid pool type: %s" % pool_type)
    return input


@POOL_LAYERS.register(name="select_adaptive_pool2d")
class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size."""

    # MARK: Magic Functions

    def __init__(
        self,
        output_size: int  = 1,
        pool_type  : str  = "fast",
        flatten    : bool = False
    ):
        super().__init__()
        # convert other falsy values to empty string for consistent TS typing
        self.pool_type = pool_type or "s"
        self.flatten   = nn.Flatten(1) if flatten else nn.Identity()
        if pool_type == "":
            self.pool = nn.Identity()  # pass through
        elif pool_type == "fast":
            if output_size != 1:
                raise ValueError()
            self.pool = FastAdaptiveAvgPool2d(flatten)
            self.flatten = nn.Identity()
        elif pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == "avgmax":
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == "catavgmax":
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        elif True:
            raise ValueError("Invalid pool type: %s" % pool_type)

    def __repr__(self):
        return (self.__class__.__name__ + " (pool_type=" + self.pool_type +
                ", flatten=" + str(self.flatten) + ")")

    # MARK: Properties

    def is_identity(self) -> bool:
        return not self.pool_type

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        input = self.pool(input)
        input = self.flatten(input)
        return input

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)


SelectAdaptivePool = SelectAdaptivePool2d
POOL_LAYERS.register(name="select_adaptive_pool", module=SelectAdaptivePool2d)
