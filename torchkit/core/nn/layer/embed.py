#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image to Patch Embedding using Conv2d. A convolution based approach to
patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn as nn
from torch import Tensor

from torchkit.core.factory import EMBED_LAYERS
from torchkit.core.type import Callable
from torchkit.core.type import Size2T
from torchkit.core.type import to_2tuple

__all__ = [
    "apply_rot_embed",
    "PatchEmbed",
    "rot",
    "RotaryEmbedding"
]


# MARK: - PatchEmbed

@EMBED_LAYERS.register(name="patch_embed")
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    # MARK: Magic Functions

    def __init__(
        self,
        img_size   : Size2T             = 224,
        patch_size : Size2T             = 16,
        in_channels: int                = 3,
        embed_dim  : int                = 768,
        norm_layer : Optional[Callable] = None,
        flatten    : bool               = True
    ):
        super().__init__()
        img_size         = to_2tuple(img_size)
        patch_size       = to_2tuple(patch_size)
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.grid_size   = (img_size[0] // patch_size[0],
                            img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten     = flatten
        self.proj        = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.norm        = (norm_layer(embed_dim) if norm_layer
                            else nn.Identity())

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        b, c, h, w = input.shape
        if h != self.img_size[0] or w != self.img_size[1]:
            raise ValueError(
                f"Input image size ({h}*{w}) doesn't match model "
                f"({self.img_size[0]}*{self.img_size[1]})."
            )
        input = self.proj(input)
        if self.flatten:
            input = input.flatten(2).transpose(1, 2)  # BCHW -> BNC
        input = self.norm(input)
        return input


# MARK: - RotaryEmbedding

def rot(input):
    return torch.stack(
        [-input[..., 1::2], input[..., ::2]], -1
    ).reshape(input.shape)


def apply_rot_embed(input: Tensor, sin_emb, cos_emb):
    return input * cos_emb + rot(input) * sin_emb


class RotaryEmbedding(nn.Module):
    """Rotary position embedding.

    This is my initial attempt at impl rotary embedding for spatial use, it
    has not been well tested, and will  likely change. It will be moved to
    its own file.

    Ffollowing impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    # MARK: Magic Functions

    def __init__(self, dim, max_freq=4):
        super().__init__()
        self.dim = dim
        self.register_buffer(
            "bands",
            2 ** torch.linspace(0., max_freq - 1, self.dim // 4),
            persistent=False
        )

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        # Assuming channel-first image where spatial dim are >= 2
        sin_emb, cos_emb = self.get_embed(input.shape[2:])
        return apply_rot_embed(input, sin_emb, cos_emb)

    def get_embed(
        self,
        shape : torch.Size,
        device: torch.device = None,
        dtype : torch.dtype  = None
    ):
        """NOTE: shape arg should include spatial dim only."""
        device = device or self.bands.device
        dtype  = dtype  or self.bands.dtype
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        N    = shape.numel()
        grid = torch.stack(
            torch.meshgrid(
                [torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=dtype)
                 for s in shape]
            ), dim=-1
        ).unsqueeze(-1)
        emb = grid * math.pi * self.bands
        sin = emb.sin().reshape(N, -1).repeat_interleave(2, -1)
        cos = emb.cos().reshape(N, -1).repeat_interleave(2, -1)
        return sin, cos
