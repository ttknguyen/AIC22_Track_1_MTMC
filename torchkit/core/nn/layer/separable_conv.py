#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depthwise Separable Conv Modules.

Basic DWS convs. Other variations of DWS exist with batch norm or activations
between the DW and PW convs such as the Depthwise modules in MobileNetV2 /
EfficientNet and Xception.
"""

from __future__ import annotations

from typing import Optional

from torch import nn as nn
from torch import Tensor

from torchkit.core.factory import CONV_LAYERS
from torchkit.core.factory import CONV_NORM_ACT_LAYERS
from torchkit.core.type import Callable
from torchkit.core.type import Padding4T
from torchkit.core.type import Size2T
from .conv import create_conv2d
from .norm_act import convert_norm_act

__all__ = [
    "SeparableConv", 
    "SeparableConv2d",
    "SeparableConvBnAct"
]


# MARK: - SeparableConvBnAct

@CONV_NORM_ACT_LAYERS.register(name="separable_conv_bn_act")
class SeparableConvBnAct(nn.Module):
    """Separable Conv w/ trailing Norm and Activation."""

    # MARK: Magic Function

    def __init__(
        self,
        in_channels       : int,
        out_channels      : int,
        kernel_size       : Size2T              = (3, 3),
        stride            : Size2T              = (1, 1),
        padding           : Optional[Padding4T] = "",
        dilation          : Size2T              = (1, 1),
        bias              : bool                = False,
        channel_multiplier: float               = 1.0,
        pw_kernel_size    : int                 = 1,
        norm_layer        : Callable            = nn.BatchNorm2d,
        act_layer         : Callable            = nn.ReLU,
        apply_act         : bool                = True,
        drop_block        : Optional[Callable]  = None
    ):
        super().__init__()
        self.conv_dw = create_conv2d(
            in_channels  = in_channels,
            out_channels = int(in_channels * channel_multiplier),
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            depthwise    = True
        )
        self.conv_pw = create_conv2d(
            in_channels  = int(in_channels * channel_multiplier),
            out_channels = out_channels,
            kernel_size  = pw_kernel_size,
            padding      = padding,
            bias         = bias
        )
        norm_act_layer = convert_norm_act(norm_layer, act_layer)
        self.bn        = norm_act_layer(out_channels, apply_act=apply_act,
                                        drop_block=drop_block)

    # MARK: Properties

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        input = self.conv_dw(input)
        input = self.conv_pw(input)
        if self.bn is not None:
            input = self.bn(input)
        return input


# MARK: - SeparableConv2d

@CONV_LAYERS.register(name="separable_conv2d")
class SeparableConv2d(nn.Module):
    """Separable Conv."""

    # MARK: Magic Function

    def __init__(
        self,
        in_channels       : int,
        out_channels      : int,
        kernel_size       : Size2T              = (3, 3),
        stride            : Size2T              = (1, 1),
        padding           : Optional[Padding4T] = "",
        dilation          : Size2T              = (1, 1),
        bias              : bool                = False,
        channel_multiplier: float               = 1.0,
        pw_kernel_size    : int                 = 1
    ):
        super().__init__()

        self.conv_dw = create_conv2d(
            in_channels  = in_channels,
            out_channels = int(in_channels * channel_multiplier),
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            depthwise    = True
        )
        self.conv_pw = create_conv2d(
            in_channels  = int(in_channels * channel_multiplier),
            out_channels = out_channels,
            kernel_size  = pw_kernel_size,
            padding      = padding,
            bias         = bias
        )

    # MARK: Properties

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        input = self.conv_dw(input)
        input = self.conv_pw(input)
        return input


SeparableConv = SeparableConv2d
CONV_LAYERS.register(name="separable_conv", module=SeparableConv)
