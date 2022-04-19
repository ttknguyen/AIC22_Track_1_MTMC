#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""EvoNormB0 (Batched) and EvoNormS0 (Sample) in PyTorch.

An attempt at getting decent performing EvoNorms running in PyTorch. While
currently faster than other impl, still quite a ways off the built-in BN in
terms of memory usage and throughput (roughly 5x mem, 1/2 - 1/3x speed).

Still very much a WIP, fiddling with buffer usage, in-place/jit
optimizations, and layouts.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from torchkit.core.factory import NORM_LAYERS
from torchkit.core.type import Callable

__all__ = [
    "EvoNormBatch", 
    "EvoNormBatch2d", 
    "EvoNormSample",
    "EvoNormSample2d"
]


# MARK: - EvoNormBatch2d

@NORM_LAYERS.register(name="evo_norm_batch2d")
class EvoNormBatch2d(nn.Module):

    # MARK: Magic Functions

    def __init__(
        self,
        num_features: int,
        apply_act   : bool               = True,
        momentum    : float              = 0.1,
        eps         : float              = 1e-5,
        drop_block  : Optional[Callable] = None
    ):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum  = momentum
        self.eps       = eps
        param_shape    = (1, num_features, 1, 1)
        self.weight = nn.Parameter(torch.ones(param_shape),  requires_grad=True)
        self.bias   = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.register_buffer("running_var", torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    # MARK: Configure

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        if input.dim() != 4:
            raise ValueError(f"Expected 4D input.")
        x_type = input.dtype
        if self.training:
            var = input.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            n   = input.numel() / input.shape[1]
            self.running_var.copy_(
                var.detach() * self.momentum * (n / (n - 1)) +
                self.running_var * (1 - self.momentum)
            )
        else:
            var = self.running_var

        if self.apply_act:
            v     = self.v.to(dtype=x_type)
            x1    = (input.var(dim=(2, 3), unbiased=False, keepdim=True) + self.eps)
            x1    = x1.sqrt().to(dtype=x_type)
            d     = input * v + x1
            d     = d.max((var + self.eps).sqrt().to(dtype=x_type))
            input = input / d
        return input * self.weight + self.bias


EvoNormBatch = EvoNormBatch2d
NORM_LAYERS.register(name="evo_norm_batch", module=EvoNormBatch)


# MARK: - EvoNormSample2d

@NORM_LAYERS.register(name="evo_norm_sample2d")
class EvoNormSample2d(nn.Module):

    # MARK: Magic Functions

    def __init__(
        self,
        num_features: int,
        apply_act   : bool               = True,
        groups      : int                = 8,
        eps         : float              = 1e-5,
        drop_block  : Optional[Callable] = None
    ):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.groups    = groups
        self.eps       = eps
        param_shape    = (1, num_features, 1, 1)
        self.weight = nn.Parameter(torch.ones(param_shape),  requires_grad=True)
        self.bias   = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.reset_parameters()

    # MARK: Configure

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        if input.dim() != 4:
            raise ValueError("Expected 4D input")
        
        b, c, h, w = input.shape
        if c % self.groups != 0:
            raise ValueError()
        
        if self.apply_act:
            n  = input * (input * self.v).sigmoid()
            input  = input.reshape(b, self.groups, -1)
            x1 = (input.var(dim=-1, unbiased=False, keepdim=True) + self.eps).sqrt()
            input  = n.reshape(b, self.groups, -1) / x1
            input  = input.reshape(b, c, h, w)
        return input * self.weight + self.bias


EvoNormSample = EvoNormSample2d
NORM_LAYERS.register(name="evo_norm_sample", module=EvoNormSample)
