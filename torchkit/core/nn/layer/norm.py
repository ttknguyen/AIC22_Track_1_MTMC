#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Normalization Layers.
"""

from __future__ import annotations

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchkit.core.factory import NORM_LAYERS
from torchkit.core.utils import console

__all__ = [
    "FractionInstanceNorm",
    "FractionInstanceNorm2d",
    "HalfGroupNorm",
    "HalfInstanceNorm",
    "HalfInstanceNorm2d",
    "HalfLayerNorm",
    "LayerNorm2d"
]


# MARK: - Register

NORM_LAYERS.register(name="batch_norm",      module=nn.BatchNorm2d)
NORM_LAYERS.register(name="batch_norm1d",    module=nn.BatchNorm1d)
NORM_LAYERS.register(name="batch_norm2d",    module=nn.BatchNorm2d)
NORM_LAYERS.register(name="batch_norm3d",    module=nn.BatchNorm3d)
NORM_LAYERS.register(name="group_norm",      module=nn.GroupNorm)
NORM_LAYERS.register(name="layer_norm",      module=nn.LayerNorm)
NORM_LAYERS.register(name="instance_norm",   module=nn.InstanceNorm2d)
NORM_LAYERS.register(name="instance_norm1d", module=nn.InstanceNorm1d)
NORM_LAYERS.register(name="instance_norm2d", module=nn.InstanceNorm2d)
NORM_LAYERS.register(name="instance_norm3d", module=nn.InstanceNorm3d)
NORM_LAYERS.register(name="sync_batch_norm", module=nn.SyncBatchNorm)


# MARK: - Group Normalization

@NORM_LAYERS.register(name="half_group_norm")
class HalfGroupNorm(nn.GroupNorm):

    # MARK: Magic Functions

    def __init__(
        self,
        num_groups  : int,
        num_channels: int,
        eps         : float = 1e-5,
        affine      : bool  = True,
        *args, **kwargs
    ):
        super().__init__(
            num_groups=num_groups, num_channels=num_channels, eps=eps,
            affine=affine, *args, **kwargs
        )

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        out_1, out_2 = torch.chunk(input, 2, dim=1)
        out_1        = F.group_norm(
            out_1, self.num_groups, self.weight, self.bias, self.eps
        )
        return torch.cat([out_1, out_2], dim=1)


# MARK: - Instance Normalization

@NORM_LAYERS.register(name="half_instance_norm2d")
class HalfInstanceNorm2d(nn.InstanceNorm2d):
    
    # MARK: Magic Functions
    
    def __init__(self, num_features: int, affine: bool = True, *args, **kwargs):
        super().__init__(
            num_features=num_features // 2, affine=affine, *args, **kwargs
        )
        
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        out_1, out_2 = torch.chunk(input, 2, dim=1)
        out_1        = F.instance_norm(
            out_1, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum,
            self.eps
        )
        return torch.cat([out_1, out_2], dim=1)


@NORM_LAYERS.register(name="fraction_instance_norm2d")
class FractionInstanceNorm2d(nn.InstanceNorm2d):
    """Perform fractional instance normalization.
    
    Args:
        num_features (int):
            Number of input features.
        alpha (float):
            Ratio of input features that will be normalized. Default: `0.5`.
        selection (str):
            Feature selection mechanism. One of: ["linear", "random",
            "interleave"]. Default: `linear`.
            - "linear"    : normalized only first half.
            - "random"    : randomly choose features to normalize.
            - "interleave": interleavingly choose features to normalize.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        num_features: int,
        alpha       : float = 0.5,
        selection   : str   = "linear",
        affine      : bool  = True,
        *args, **kwargs
    ):
        self.in_channels = num_features
        self.alpha       =  alpha
        self.selection   = selection
        super().__init__(
            num_features=math.ceil(num_features * self.alpha), affine=affine,
            *args, **kwargs
        )

        if self.selection not in ["linear", "random", "interleave"]:
            raise ValueError()
 
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        _, c, _, _ = input.shape
        
        if self.alpha == 0.0:
            return input
        elif self.alpha == 1.0:
            return F.instance_norm(
                input, self.running_mean, self.running_var, self.weight,
                self.bias, self.training or not self.track_running_stats,
                self.momentum, self.eps
            )
        else:
            if self.selection == "random":
                out1_idxes = random.sample(range(self.in_channels), self.num_features)
                out2_idxes = list(set(range(self.in_channels)) - set(out1_idxes))
                out1_idxes = Tensor(out1_idxes).to(torch.int).to(input.device)
                out2_idxes = Tensor(out2_idxes).to(torch.int).to(input.device)
                out1       = torch.index_select(input, 1, out1_idxes)
                out2       = torch.index_select(input, 1, out2_idxes)
            elif self.selection == "interleave":
                skip       = int(math.floor(self.in_channels / self.num_features))
                out1_idxes = []
                for i in range(0, self.in_channels, skip):
                    if len(out1_idxes) < self.num_features:
                        out1_idxes.append(i)
                out2_idxes = list(set(range(self.in_channels)) - set(out1_idxes))
                # print(len(out1_idxes), len(out2_idxes), self.num_features)
                out1_idxes = Tensor(out1_idxes).to(torch.int).to(input.device)
                out2_idxes = Tensor(out2_idxes).to(torch.int).to(input.device)
                out1       = torch.index_select(input, 1, out1_idxes)
                out2       = torch.index_select(input, 1, out2_idxes)
            else:  # Half-Half
                split_size = [self.num_features, c - self.num_features]
                out1, out2 = torch.split(input, split_size, dim=1)
            
            out1 = F.instance_norm(
                out1, self.running_mean, self.running_var, self.weight,
                self.bias, self.training or not self.track_running_stats,
                self.momentum, self.eps
            )
            return torch.cat([out1, out2], dim=1)
    

HalfInstanceNorm = HalfInstanceNorm2d
NORM_LAYERS.register(name="half_instance_norm", module=HalfInstanceNorm)

FractionInstanceNorm = FractionInstanceNorm2d
NORM_LAYERS.register(name="fraction_instance_norm", module=HalfInstanceNorm)


# MARK: - Layer Normalization

@NORM_LAYERS.register(name="layer_norm2d")
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of `2D` spatial [B, C, H, W] tensors."""

    # MARK: Magic Functions

    def __init__(self, num_channels: int, *args, **kwargs):
        super().__init__(num_channels, *args, **kwargs)

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
			self.bias, self.eps
        ).permute(0, 3, 1, 2)


@NORM_LAYERS.register(name="half_layer_norm")
class HalfLayerNorm(nn.LayerNorm):

    # MARK: Magic Functions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        out_1, out_2 = torch.chunk(input, 2, dim=1)
        out_1        = F.layer_norm(
            out_1, self.normalized_shape, self.weight, self.bias, self.eps
        )
        return torch.cat([out_1, out_2], dim=1)
