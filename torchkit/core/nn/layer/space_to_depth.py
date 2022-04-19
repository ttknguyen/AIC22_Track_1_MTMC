#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "DepthToSpace", 
    "SpaceToDepth", 
    "SpaceToDepthJit", 
    "SpaceToDepthModule",
]


# MARK: - SpaceToDepth

class SpaceToDepth(nn.Module):

    # MARK: Magic Functions

    def __init__(self, block_size: int = 4):
        super().__init__()
        if block_size != 4:
            raise ValueError()
        self.bs = block_size

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        N, C, H, W = input.size()
        input = input.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        input = input.permute(0, 3, 5, 1, 2, 4).contiguous()                    # (N, bs, bs, C, H//bs, W//bs)
        input = input.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)   # (N, C*bs^2, H//bs, W//bs)
        return input


# MARK: - SpaceToDepthJit

@torch.jit.script
class SpaceToDepthJit(object):

    # MARK: Magic Functions

    def __call__(self, input: Tensor):
        # assuming hard-coded that block_size==4 for acceleration
        N, C, H, W = input.size()
        input = input.view(N, C, H // 4, 4, W // 4, 4)        # (N, C, H//bs, bs, W//bs, bs)
        input = input.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        input = input.view(N, C * 16, H // 4, W // 4)         # (N, C*bs^2, H//bs, W//bs)
        return input


# MARK: - SpaceToDepthModule

class SpaceToDepthModule(nn.Module):

    # MARK: Magic Functions

    def __init__(self, no_jit: bool = False):
        super().__init__()
        if not no_jit:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepth()

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        return self.op(input)


# MARK: - DepthToSpace

class DepthToSpace(nn.Module):

    # MARK: Magic Functions

    def __init__(self, block_size: int):
        super().__init__()
        self.bs = block_size

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        N, C, H, W = input.size()
        input = input.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)    # (N, bs, bs, C//bs^2, H, W)
        input = input.permute(0, 3, 4, 1, 5, 2).contiguous()                  # (N, C//bs^2, H, bs, W, bs)
        input = input.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return input
