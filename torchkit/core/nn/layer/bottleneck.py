#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bottleneck Layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from torchkit.core.factory import BOTTLENECK_LAYERS
from torchkit.core.type import Size2T
from .act import Mish
from .conv_norm_act import ConvBnMish
from .conv_norm_act import GhostConv
from .dw_conv import DWConv

__all__ = [
    "Bottleneck",
    "BottleneckCSP",
    "BottleneckCSP2",
    "GhostBottleneck",
    "VoVCSP"
]


# MARK: - Bottleneck

@BOTTLENECK_LAYERS.register(name="bottleneck")
class Bottleneck(nn.Module):
    """Standard bottleneck.
    
    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        shortcut (bool):
            Use shortcut connection?. Default: `True`.
        groups (int):
            Default: `1`.
        expansion (float):
            Default: `0.5`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        shortcut    : bool  = True,
        groups      : int   = 1,
        expansion   : float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # Hidden channels
        self.conv1 = ConvBnMish(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBnMish(hidden_channels, out_channels, 3, 1,
                                groups=groups)
        self.add   = shortcut and in_channels == out_channels
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        pred = ((input + self.conv2(self.conv1(input))) if self.add
                else self.conv2(self.conv1(input)))
        return pred


# MARK: - BottleneckCSP

@BOTTLENECK_LAYERS.register(name="bottleneck_csp")
class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks

    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        number (int):
            Number of bottleneck layers to use. Default: `1`.
        shortcut (bool):
            Use shortcut connection?. Default: `True`.
        groups (int):
            Default: `1`.
        expansion (float):
            Default: `0.5`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        number      : int   = 1,
        shortcut    : bool  = True,
        groups      : int   = 1,
        expansion   : float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # Hidden channels
        self.conv1 = ConvBnMish(in_channels, hidden_channels, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, (1, 1), (1, 1),
                               bias=False)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, (1, 1),
                               (1, 1), bias=False)
        self.conv4 = ConvBnMish(2 * hidden_channels, out_channels, 1, 1)
        # Applied to cat(cv2, cv3)
        self.bn    = nn.BatchNorm2d(2 * hidden_channels)
        self.act   = Mish()
        self.m     = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels, shortcut, groups,
                         expansion=1.0)
              for _ in range(number)]
        )
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        y1   = self.conv3(self.m(self.conv1(input)))
        y2   = self.conv2(input)
        pred = self.conv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
        return pred


# MARK: - BottleneckCSP2

@BOTTLENECK_LAYERS.register(name="bottleneck_csp2")
class BottleneckCSP2(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks

    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        number (int):
            Number of bottleneck layers to use. Default: `1`.
        groups (int):
            Default: `1`.
        expansion (float):
            Default: `0.5`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        number      : int   = 1,
        shortcut    : bool  = False,
        groups      : int   = 1,
        expansion   : float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels)  # Hidden channels
        self.conv1 = ConvBnMish(in_channels, hidden_channels, 1, 1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, (1, 1), (1, 1),
                               bias=False)
        self.conv3 = ConvBnMish(2 * hidden_channels, out_channels, 1, 1)
        self.bn    = nn.BatchNorm2d(2 * hidden_channels)
        self.act   = Mish()
        self.m     = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels, shortcut,
                         groups, expansion=1.0)
              for _ in range(number)]
        )
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        x1 = self.conv1(input)
        y1 = self.m(x1)
        y2 = self.conv2(x1)
        return self.conv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))


# MARK: - GhostBottleneck

@BOTTLENECK_LAYERS.register(name="ghost_bottleneck")
class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T,
        stride      : Size2T,
    ):
        super().__init__()
        c_ = out_channels // 2
        self.conv = nn.Sequential(
            GhostConv(in_channels, c_, 1, 1),     # pw
            (DWConv(c_, c_, kernel_size, stride, apply_act=False)
             if stride == 2 else nn.Identity()),  # dw
            GhostConv(c_, out_channels, 1, 1, apply_act=False)
        )  # pw-linear
        self.shortcut = nn.Sequential(
            DWConv(in_channels, in_channels, kernel_size, stride,
                   apply_act=False),
            ConvBnMish(in_channels, in_channels, 1, 1, apply_act=False)
        ) if stride == 2 else nn.Identity()
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input) + self.shortcut(input)
    

# MARK: - VoVCSP

@BOTTLENECK_LAYERS.register(name="vov_csp")
class VoVCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks

    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        number (int):
            Number of bottleneck layers to use.
        groups (int):
            Default: `1`.
        expansion (float):
            Default: `0.5`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        number      : int   = 1,
        shortcut    : bool  = True,
        groups      : int   = 1,
        expansion   : float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels)  # Hidden channels
        self.conv1 = ConvBnMish(in_channels // 2, hidden_channels // 2,
                            kernel_size=3, stride=1)
        self.conv2 = ConvBnMish(hidden_channels // 2, hidden_channels // 2,
                            kernel_size=3, stride=1)
        self.conv3 = ConvBnMish(hidden_channels, out_channels, kernel_size=1,
                            stride=1)
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        _, x1 = input.chunk(2, dim=1)
        x1    = self.conv1(x1)
        x2    = self.conv2(x1)
        pred  = self.conv3(torch.cat((x1, x2), dim=1))
        return pred
