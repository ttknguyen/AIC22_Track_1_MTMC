#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Drop Layers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from torchkit.core.factory import DROP_LAYERS

__all__ = [
    "drop_block_2d",
    "drop_block_fast_2d",
    "drop_path",
    "DropBlock",
    "DropBlock2d",
    "Dropout",
    "DropPath"
]


# MARK: - DropBlock

def drop_block_2d(
    input      : Tensor,
    drop_prob  : float = 0.1,
    block_size : int   = 7,
    gamma_scale: float = 1.0,
    with_noise : bool  = False,
    inplace    : bool  = False,
    batchwise  : bool  = False
) -> Tensor:
    """DropBlock with an experimental gaussian noise option. This layer has
    been tested on a few training runs with success, but needs further
    validation and possibly optimization for lower runtime impact.
    
    Papers:
    `DropBlock: A regularization method for convolutional networks`
    (https://arxiv.org/abs/1810.12890)
    """
    b, c, h, w         = input.shape
    total_size         = w * h
    clipped_block_size = min(block_size, min(w, h))
    # seed_drop_rate, the gamma parameter
    gamma = (gamma_scale * drop_prob * total_size / clipped_block_size ** 2 /
             ((w - block_size + 1) * (h - block_size + 1)))

    # Forces the block to be inside the feature map.
    w_i, h_i    = torch.meshgrid(torch.arange(w).to(input.device),
                                 torch.arange(h).to(input.device))
    valid_block = (
        ((w_i >= clipped_block_size // 2) &
         (w_i < w - (clipped_block_size - 1) // 2)) &
        ((h_i >= clipped_block_size // 2) &
         (h_i < h - (clipped_block_size - 1) // 2))
    )
    valid_block = torch.reshape(valid_block, (1, 1, h, w)).to(dtype=input.dtype)

    if batchwise:
        # One mask for whole batch, quite a bit faster
        uniform_noise = torch.rand((1, c, h, w), dtype=input.dtype,
                                   device=input.device)
    else:
        uniform_noise = torch.rand_like(input)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1)
    block_mask = block_mask.to(dtype=input.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size = clipped_block_size,
        # block_size,
        stride      = 1,
        padding     = clipped_block_size // 2
    )

    if with_noise:
        normal_noise = (
            torch.randn((1, c, h, w), dtype=input.dtype, device=input.device)
            if batchwise else torch.randn_like(input)
        )
        if inplace:
            input.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            input = input * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (block_mask.numel() /
                           block_mask.to(dtype=torch.float32).sum().add(1e-7))
        normalize_scale = normalize_scale.to(input.dtype)
        if inplace:
            input.mul_(block_mask * normalize_scale)
        else:
            input = input * block_mask * normalize_scale
    return input


def drop_block_fast_2d(
    input      : Tensor,
    drop_prob  : float = 0.1,
    block_size : int   = 7,
    gamma_scale: float = 1.0,
    with_noise : bool  = False,
    inplace    : bool  = False,
    batchwise  : bool  = False
) -> Tensor:
    """DropBlock with an experimental gaussian noise option. Simplied from
    above without concern for valid block mask at edges.

    Papers:
    `DropBlock: A regularization method for convolutional networks`
    (https://arxiv.org/abs/1810.12890)
    """
    b, c, h, w 		   = input.shape
    total_size		   = w * h
    clipped_block_size = min(block_size, min(w, h))
    gamma = (gamma_scale * drop_prob * total_size / clipped_block_size ** 2 /
             ((w - block_size + 1) * (h - block_size + 1)))

    if batchwise:
        # One mask for whole batch, quite a bit faster
        block_mask = torch.rand((1, c, h, w), dtype=input.dtype, device=input.device)
        block_mask = block_mask < gamma
    else:
        # Mask per batch element
        block_mask = torch.rand_like(input) < gamma
    block_mask = F.max_pool2d(
        block_mask.to(input.dtype), kernel_size=clipped_block_size, stride=1,
        padding=clipped_block_size // 2
    )

    if with_noise:
        normal_noise = (
            torch.randn((1, c, h, w), dtype=input.dtype, device=input.device)
            if batchwise else torch.randn_like(input)
        )
        if inplace:
            input.mul_(1. - block_mask).add_(normal_noise * block_mask)
        else:
            input = input * (1. - block_mask) + normal_noise * block_mask
    else:
        block_mask 	    = 1 - block_mask
        normalize_scale = (block_mask.numel() /
                           block_mask.to(dtype=torch.float32).sum().add(1e-7))
        normalize_scale = normalize_scale.to(dtype=input.dtype)
        if inplace:
            input.mul_(block_mask * normalize_scale)
        else:
            input = input * block_mask * normalize_scale
    return input


@DROP_LAYERS.register(name="drop_block2d")
class DropBlock2d(nn.Module):
    """DropBlock."""

    # MARK: Magic Functions

    def __init__(
        self,
        drop_prob  : float = 0.1,
        block_size : int   = 7,
        gamma_scale: float = 1.0,
        with_noise : bool  = False,
        inplace    : bool  = False,
        batchwise  : bool  = False,
        fast       : bool  = True
    ):
        super().__init__()
        self.drop_prob   = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size  = block_size
        self.with_noise  = with_noise
        self.inplace     = inplace
        self.batchwise   = batchwise
        self.fast        = fast  # FIXME finish comparisons of fast vs not

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or not self.drop_prob:
            return input
        if self.fast:
            return drop_block_fast_2d(
                input, self.drop_prob, self.block_size, self.gamma_scale,
				self.with_noise, self.inplace, self.batchwise
            )
        else:
            return drop_block_2d(
                input, self.drop_prob, self.block_size, self.gamma_scale,
				self.with_noise, self.inplace, self.batchwise
            )


DropBlock = DropBlock2d
DROP_LAYERS.register(name="drop_block", module=DropBlock)


# MARK: - Dropout

@DROP_LAYERS.register(name="dropout")
class Dropout(nn.Dropout):
    """A wrapper for `torch.nn.Dropout`, We rename the `p` of
    `torch.nn.Dropout` to `drop_prob` so as to be consistent with `DropPath`.

    Args:
        drop_prob (float):
            Probability of the elements to be zeroed. Default: `0.5`.
        inplace (bool):
            Do the operation inplace or not. Default: `False`.
    """

    # MARK: Magic Functions

    def __init__(self, drop_prob: float = 0.5, inplace: bool = False):
        super().__init__(p=drop_prob, inplace=inplace)


# MARK: - DropPath

def drop_path(
    input: Tensor, drop_prob: float = 0.0, training: bool = False
) -> Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks). We follow the implementation:
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py

    Args:
        input (Tensor):
            Input image.
        drop_prob (float):
            Probability of the path to be zeroed. Default: `0.0`.
        training (bool):
            Is in training run?. Default: `False`.

    Returns:
        output (Tensor):
            Output image.
    """
    if drop_prob == 0.0 or not training:
        return input
    
    # NOTE: Handle tensors with different dimensions, not just 4D tensors.
    keep_prob = 1 - drop_prob
    shape	  = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = (keep_prob +
                     torch.rand(shape, dtype=input.dtype, device=input.device))
    output 		  = input.div(keep_prob) * random_tensor.floor()
    return output


@DROP_LAYERS.register(name="drop_path")
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.
    
    Attributes:
        drop_prob (float):
            Probability of the path to be zeroed. Default: `0.1`.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        return drop_path(
            input=input, drop_prob=self.drop_prob, training=self.training
        )


# MARK: - Dropout

@DROP_LAYERS.register(name="dropout")
class Dropout(nn.Dropout):
    """A wrapper for `torch.nn.Dropout`, We rename the `p` of
    `torch.nn.Dropout` to `drop_prob` so as to be consistent with `DropPath`.
    
    Args:
        drop_prob (float):
            Probability of the elements to be zeroed. Default: `0.5`.
        inplace (bool):
            Do the operation inplace or not. Default: `False`.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, drop_prob: float = 0.5, inplace: bool = False):
        super().__init__(p=drop_prob, inplace=inplace)
