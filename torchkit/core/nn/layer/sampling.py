#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Downsampling and upsampling layers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from torchkit.core.factory import SAMPLING_LAYERS
from torchkit.core.type import Size2T

__all__ = [
    "Downsample",
    "Downsample2",
    "Downsampling",
    "Downsampling2",
    "PixelShufflePack",
    "SkipUpsample",
    "SkipUpsampling",
    "Upsample", 
    "Upsampling"
]


# MARK: - Downsample/Downsampling

@SAMPLING_LAYERS.register(name="downsample")
class Downsample(nn.Sequential):
    """
    
    Args:
        in_channels (int):
            Number of input channels.
        scale_factor (int):
            Fscale factor. Default: `0`.
        mode (str, optional):
            Fupsampling algorithm. One of: [`nearest`, `linear`, `bilinear`,
            `bicubic`, `trilinear`]. Default: `nearest`.
        align_corners (bool, optional):
            If `True`, the corner pixels of the input and output tensors are
            aligned, and thus preserving the values at those pixels. This
            only has effect when :attr:`mode` is `linear`, `bilinear`, or
            `trilinear`. Default: `False`.
    """
    
    # MARK: Magic Functions

    def __init__(
        self,
        in_channels  : int,
        scale_factor : int            = 0,
        mode         : str            = "bilinear",
        align_corners: Optional[bool] = False
    ):
        super().__init__()
        self.add_module(
            "upsample", nn.Upsample(
                scale_factor=0.5, mode=mode, align_corners=align_corners
            )
        )
        self.add_module(
            "conv", nn.Conv2d(
                in_channels, in_channels + scale_factor, kernel_size=(1, 1),
                stride=(1, 1), padding=0, bias=False
            )
        )


@SAMPLING_LAYERS.register(name="downsample2")
class Downsample2(nn.Module):
    """http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        n_planes     : int,
        factor       : int,
        kernel_type  : str,
        phase        : float           = 0.0,
        kernel_width : Optional[int]   = None,
        support      : Optional[int]   = None,
        sigma        : Optional[float] = None,
        preserve_size: bool            = False
    ):
        super().__init__()

        if phase not in [0, 0.5]:
            raise ValueError("phase should be 0 or 0.5")

        if kernel_type == "lanczos2":
            support      = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = "lanczos"
        elif kernel_type == "lanczos3":
            support      = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = "lanczos"
        elif kernel_type == "gauss12":
            kernel_width = 7
            sigma        = 1 / 2
            kernel_type_ = "gauss"
        elif kernel_type == "gauss1sq2":
            kernel_width = 9
            sigma        = 1.0 / np.sqrt(2)
            kernel_type_ = "gauss"
        elif kernel_type in ["lanczos", "gauss", "box"]:
            kernel_type_ = kernel_type
        else:
            raise ValueError("wrong name kernel")

        # Note that `kernel width` will be different to actual size for phase = 1/2
        self.kernel = get_kernel(
            factor, kernel_type_, phase, kernel_width, support=support,
            sigma=sigma
        )

        downsampler = nn.Conv2d(
            n_planes, n_planes, kernel_size=self.kernel.shape,
	        stride=(factor, factor), padding=0
        )
        downsampler.weight.data[:] = 0
        downsampler.bias.data[:]   = 0

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            downsampler.weight.data[i, i] = kernel_torch

        self.downsampler_ = downsampler

        if preserve_size:
            if self.kernel.shape[0] % 2 == 1:
                pad = int((self.kernel.shape[0] - 1) / 2.)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.)

            self.padding = nn.ReplicationPad2d(pad)

        self.preserve_size = preserve_size
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        if self.preserve_size:
            x = self.padding(input)
        else:
            x = input
        return self.downsampler_(x)


def get_kernel(
	factor      : int,
	kernel_type : str,
	phase       : float,
	kernel_width: int,
	support     : Optional[int]   = None,
	sigma       : Optional[float] = None
) -> np.ndarray:
    if kernel_type not in ["lanczos", "gauss", "box"]:
        raise ValueError()

    if phase == 0.5 and kernel_type != "box":
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])
    
    # NOTE: Box
    if kernel_type == "box":
        if phase != 0.5:
            raise ValueError("Box filter is always half-phased")
        kernel[:] = 1.0 / (kernel_width * kernel_width)
    # NOTE: Gauss
    elif kernel_type == "gauss":
        if sigma is None:
            raise ValueError("sigma is not specified")
        if phase == 0.5:
            raise ValueError("phase 1/2 for gauss not implemented")

        center   = (kernel_width + 1.0) / 2.0
        sigma_sq = sigma * sigma

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di                   = (i - center) / 2.0
                dj                   = (j - center) / 2.0
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2.0 * np.pi * sigma_sq)
    # NOTE: Lanczos
    elif kernel_type == "lanczos":
        if support is None:
            raise ValueError("support is not specified")
        center = (kernel_width + 1) / 2.

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor
                    dj = abs(j + 0.5 - center) / factor
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor

                pi_sq = np.pi * np.pi

                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)
                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)

                kernel[i - 1][j - 1] = val
    else:
        raise ValueError("wrong method name")

    kernel /= kernel.sum()
    return kernel


Downsampling  = Downsample
Downsampling2 = Downsample2
SAMPLING_LAYERS.register(name="downsampling",  module=Downsampling)
SAMPLING_LAYERS.register(name="downsampling2", module=Downsampling2)


# MARK: - Upsample/Upsampling

@SAMPLING_LAYERS.register(name="upsample")
class Upsample(nn.Sequential):
    """

    Args:
        in_channels (int):
            Number of input channels.
        scale_factor (int):
            Fscale factor. Default: `0`.
        mode (str, optional):
            Fupsampling algorithm. One of: [`nearest`, `linear`, `bilinear`,
            `bicubic`, `trilinear`]. Default: `nearest`.
        align_corners (bool, optional):
            If `True`, the corner pixels of the input and output tensors are
            aligned, and thus preserving the values at those pixels. This
            only has effect when :attr:`mode` is `linear`, `bilinear`, or
            `trilinear`. Default: `False`.
    """
    
    # MARK: Magic Functions

    def __init__(
        self,
        in_channels  : int,
        scale_factor : int            = 0,
        mode         : str            = "bilinear",
        align_corners: Optional[bool] = False
    ):
        
        super().__init__()
        self.add_module(
            "upsample", nn.Upsample(
                scale_factor=2.0, mode=mode, align_corners=align_corners
            )
        )
        self.add_module(
            "conv", nn.Conv2d(
                in_channels + scale_factor, in_channels, kernel_size=(1, 1),
                stride=(1, 1), padding=0, bias=False)
        )


Upsampling = Upsample
SAMPLING_LAYERS.register(name="upsampling", module=Upsampling)


# MARK: - SkipUpsample/SkipUpsampling

@SAMPLING_LAYERS.register(name="skip_upsample")
class SkipUpsample(nn.Module):
    """

    Args:
        in_channels (int):
            Number of input channels.
        scale_factor (int):
            Fscale factor. Default: `0`.
        mode (str, optional):
            Fupsampling algorithm. One of: [`nearest`, `linear`, `bilinear`,
            `bicubic`, `trilinear`]. Default: `nearest`.
        align_corners (bool, optional):
            If `True`, the corner pixels of the input and output tensors are
            aligned, and thus preserving the values at those pixels. This
            only has effect when :attr:`mode` is `linear`, `bilinear`, or
            `trilinear`. Default: `False`.
    """
    
    # MARK: Magic Functions

    def __init__(
        self,
        in_channels  : int,
        scale_factor : int            = 0,
        mode         : str            = "bilinear",
        align_corners: Optional[bool] = False
    ):
        
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode=mode,
                        align_corners=align_corners),
            nn.Conv2d(in_channels + scale_factor, in_channels,
                      kernel_size=(1, 1),  stride=(1, 1), padding=0, bias=False)
        )
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor, skip: Tensor) -> Tensor:
        """Forward pass.
		
		Args:
			input (Tensor):
				Input image.
			skip (Tensor):
				Fskip connection image.
				
		Returns:
			pred (Tensor):
				Output image.
		"""
        pred  = self.up(input)
        pred += skip
        return pred


SkipUpsampling = SkipUpsample
SAMPLING_LAYERS.register(name="skip_upsampling", module=SkipUpsampling)


# MARK: - PixelShufflePack

@SAMPLING_LAYERS.register(name="pixel_shuffle")
class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer. This module packs `F.pixel_shuffle()`
    and a nn.Conv2d module together to achieve a simple upsampling with pixel
    shuffle.
    
    Args:
        in_channels (int):
            Number of input channels.
        out_channels (int):
            Number of output channels.
        scale_factor (int):
            Upsample ratio.
        upsample_kernel (int, tuple):
            Kernel size of the conv layer to expand the channels.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        scale_factor   : int,
        upsample_kernel: Size2T,
    ):
        
        super().__init__()
        self.upsample_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels * scale_factor * scale_factor,
            kernel_size  = upsample_kernel,
            padding      = (upsample_kernel - 1) // 2
        )
        self.init_weights()

    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        pred = self.upsample_conv(input)
        pred = F.pixel_shuffle(pred, self.scale_factor)
        return pred
