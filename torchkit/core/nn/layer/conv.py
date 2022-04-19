#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convolution Layers.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from torchkit.core.factory import CONV_LAYERS
from torchkit.core.type import Padding2T
from torchkit.core.type import Padding4T
from torchkit.core.type import Size2T
from torchkit.core.type import to_2tuple
from .padding import get_padding_value
from .padding import pad_same

__all__ = [
    "_split_channels", 
    "CondConv", 
    "CondConv2d", 
    "conv2d_same", 
    "Conv2dSame",
    "Conv2dTF", 
    "ConvSame",
    "ConvTF", 
    "create_conv2d",
    "create_conv2d_pad",
    "get_condconv_initializer",
    "MixedConv",
    "MixedConv2d"
]


# MARK: - Register

CONV_LAYERS.register(name="conv",   module=nn.Conv2d)
CONV_LAYERS.register(name="conv1d", module=nn.Conv1d)
CONV_LAYERS.register(name="conv2d", module=nn.Conv2d)
CONV_LAYERS.register(name="conv3d", module=nn.Conv3d)


# MARK: - CondConv2d

def get_condconv_initializer(initializer, num_experts: int, expert_shape):
    def condconv_initializer(weight: Tensor):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if (
            len(weight.shape) != 2 or
            weight.shape[0] != num_experts or
            weight.shape[1] != num_params
        ):
            raise (ValueError("CondConv variables must have shape "
                              "[num_experts, num_params]"))
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))
    return condconv_initializer


@CONV_LAYERS.register(name="cond_conv2d")
class CondConv2d(nn.Module):
    """Conditionally Parameterized Convolution. Inspired by:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py

    Grouped convolution hackery for parallel execution of the per-sample kernel
    filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """

    __constants__ = ["in_channels", "out_channels", "dynamic_padding"]

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T              = (3, 3),
        stride      : Size2T              = (1, 1),
        padding     : Optional[Padding4T] = "",
        dilation    : Size2T              = (1, 1),
        groups      : int                 = 1,
        bias        : Optional[bool]      = False,
        num_experts : int                 = 4
    ):
        super(CondConv2d, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = to_2tuple(kernel_size)
        self.stride       = to_2tuple(stride)

        padding_val, is_padding_dynamic = get_padding_value(
			padding, kernel_size, stride=stride, dilation=dilation
		)
        # if in forward to work with torchscript
        self.dynamic_padding = is_padding_dynamic
        self.padding         = to_2tuple(padding_val)
        self.dilation        = to_2tuple(dilation)
        self.groups          = groups
        self.num_experts     = num_experts

        self.weight_shape = (
			(self.out_channels, self.in_channels // self.groups) +
			self.kernel_size
		)
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(
			Tensor(self.num_experts, weight_num_param)
		)

        if bias:
            self.bias_shape = (self.out_channels,)
            self.bias = torch.nn.Parameter(
				Tensor(self.num_experts, self.out_channels)
			)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    # MARK: Configure

    def reset_parameters(self):
        init_weight = get_condconv_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)),
			self.num_experts, self.weight_shape
        )
        init_weight(self.weight)
        if self.bias is not None:
            fan_in    = np.prod(self.weight_shape[1:])
            bound     = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                partial(nn.init.uniform_, a=-bound, b=bound), self.num_experts,
				self.bias_shape
            )
            init_bias(self.bias)

    # MARK: Forward Pass

    def forward(self, input: Tensor, routing_weights: Tensor) -> Tensor:
        b, c, h, w = input.shape
        weight     = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (
			(b * self.out_channels, self.in_channels // self.groups) +
			self.kernel_size
		)
        weight = weight.view(new_weight_shape)
        bias   = None

        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(b * self.out_channels)
        # Move batch elements with channels so each batch element can be
		# efficiently convolved with separate kernel
        input = input.view(1, b * c, h, w)
        if self.dynamic_padding:
            out = conv2d_same(
                input, weight, bias, stride=self.stride, padding=self.padding,
				dilation=self.dilation, groups=self.groups * b
            )
        else:
            out = F.conv2d(
                input, weight, bias, stride=self.stride, padding=self.padding,
				dilation=self.dilation, groups=self.groups * b
            )
        out = out.permute([1, 0, 2, 3]).view(
            b, self.out_channels, out.shape[-2], out.shape[-1]
        )

        # Literal port (from TF definition)
        # input = torch.split(input, 1, 0)
        # weight = torch.split(weight, 1, 0)
        # if self.bias is not None:
        #     bias = torch.matmul(routing_weights, self.bias)
        #     bias = torch.split(bias, 1, 0)
        # else:
        #     bias = [None] * B
        # out = []
        # for xi, wi, bi in zip(input, weight, bias):
        #     wi = wi.view(*self.weight_shape)
        #     if bi is not None:
        #         bi = bi.view(*self.bias_shape)
        #     out.append(self.conv_fn(
        #         xi, wi, bi, stride=self.stride, padding=self.padding,
        #         dilation=self.dilation, groups=self.groups))
        # out = torch.cat(out, 0)
        return out


CondConv = CondConv2d
CONV_LAYERS.register(name="cond_conv", module=CondConv)


# MARK: - Conv2dTF

@CONV_LAYERS.register(name="conv2d_tf")
class Conv2dTF(nn.Conv2d):
    """Implementation of 2D convolution in TensorFlow with `padding` as "same",
    which applies padding to input (if needed) so that input image gets fully
    covered by filter and stride you specified. For stride `1`, this will
    ensure that output image size is same as input. For stride of 2, output
    dimensions will be half, for example.
    
    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (Size2T):
            Size of the convolving kernel
        stride (Size2T):
            Stride of the convolution. Default: `1`.
        padding (Padding2T, optional):
            Zero-padding added to both sides of the input. Default: `0`.
        dilation (str, Size2T, optional):
            Spacing between kernel elements. Default: `1`.
        groups (int):
            Number of blocked connections from input channels to output
            channels. Default: `1`.
        bias (bool):
            If `True`, adds a learnable bias to the output. Default: `True`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T,
        stride      : Size2T              = 1,
        padding     : Optional[Padding2T] = 0,
        dilation    : Size2T              = 1,
        groups      : int                 = 1,
        bias        : bool                = True,
        *args, **kwargs
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, *args, **kwargs
        )
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            input (Tensor):
                Input image.

        Returns:
            pred (Tensor):
                Output image.
        """
        img_h, img_w       = input.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h    = max((output_h - 1) * self.stride[0] + (kernel_h - 1) *
                       self.dilation[0] + 1 - img_h, 0)
        pad_w    = max((output_w - 1) * self.stride[1] + (kernel_w - 1) *
                       self.dilation[1] + 1 - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            input = F.pad(input, [pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2])
        pred = F.conv2d(input, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
        return pred


ConvTF = Conv2dTF
CONV_LAYERS.register(name="conv_tf", module=ConvTF)


# MARK: - Conv2dSame

def conv2d_same(
    input       : Tensor,
    weight  : Tensor,
    bias    : Optional[Tensor] = None,
    stride  : Size2T                 = (1, 1),
    padding : Optional[Padding4T]    = 0,
    dilation: Size2T                 = (1, 1),
    groups  : int                    = 1,
    **_
):
    """Functional interface for Same Padding Convolution 2D.

    Args:
        input (Tensor):
            Input image.
        weight (Tensor):
            Weight.
        bias (Tensor, optional):
            Bias value.
        stride (Size2T):
            Stride of the convolution. Default: `(1, 1)`.
        padding (Padding4T, optional):
            Zero-padding added to both sides of the input. Default: `0`.
        dilation (Size2T):
            Spacing between kernel elements. Default: `(1, 1)`.
        groups (int):
            Number of blocked connections from input channels to output
            channels. Default: `1`.

    Returns:
        input (Tensor):
            Output image.
    """
    input = pad_same(input, weight.shape[-2:], stride, dilation)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


@CONV_LAYERS.register(name="conv2d_same")
class Conv2dSame(nn.Conv2d):
    """Tensorflow like `SAME` convolution wrapper for 2D convolutions.

    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (Size2T):
            Size of the convolving kernel. Default: `(1, 1)`.
        stride (Size2T):
            Stride of the convolution. Default: `(1, 1)`.
        padding (Padding4T, optional):
            Zero-padding added to both sides of the input. Default: `0`.
        dilation (Size2T):
            Spacing between kernel elements. Default: `(1, 1)`.
        groups (int):
            Default: `1`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T              = (1, 1),
        stride      : Size2T              = (1, 1),
        padding     : Optional[Padding4T] = 0,
        dilation    : Size2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = True,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias)

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        return conv2d_same(input, self.weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)


ConvSame = Conv2dSame
CONV_LAYERS.register(name="conv_same", module=ConvSame)


# MARK: - MixedConv2d

def _split_channels(num_channels: int, num_groups: int):
    split     = [num_channels // num_groups for _ in range(num_groups)]
    split[0] += num_channels - sum(split)
    return split


@CONV_LAYERS.register(name="mixed_conv2d")
class MixedConv2d(nn.ModuleDict):
    """Mixed Convolution from the paper `MixConv: Mixed Depthwise
    Convolutional Kernels` (https://arxiv.org/abs/1907.09595)

    Based on MDConv and GroupedConv in MixNet implementation:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T              = (3, 3),
        stride      : Size2T              = (1, 1),
        padding     : Optional[Padding4T] = "",
        dilation    : Size2T              = (1, 1),
        depthwise   : bool                = False,
        **kwargs
    ):
        super().__init__()
        kernel_size       = kernel_size
        stride            = to_2tuple(stride)
        dilation          = to_2tuple(dilation)
        num_groups        = len(kernel_size)
        in_splits         = _split_channels(in_channels, num_groups)
        out_splits        = _split_channels(out_channels, num_groups)
        self.in_channels  = sum(in_splits)
        self.out_channels = sum(out_splits)

        for idx, (k, in_ch, out_ch) in enumerate(
            zip(kernel_size, in_splits, out_splits)
        ):
            conv_groups = in_ch if depthwise else 1
            # Use add_module to keep key space clean
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride, padding=padding,
                    dilation=dilation, groups=conv_groups, **kwargs
                )
            )
        self.splits = in_splits

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        x_split = torch.split(input, self.splits, 1)
        x_out   = [c(x_split[i]) for i, c in enumerate(self.values())]
        input   = torch.cat(x_out, 1)
        return input


MixedConv = MixedConv2d
CONV_LAYERS.register(name="mixed_conv", module=MixedConv)


# MARK: - Builder

def create_conv2d_pad(
    in_channels: int, out_channels: int, kernel_size: Size2T, **kwargs
) -> nn.Conv2d:
    """Create 2D Convolution layer with padding."""
    padding = kwargs.pop("padding", "")
    kwargs.setdefault("bias", False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)

    if is_dynamic:
        return Conv2dSame(in_channels, out_channels, kernel_size, **kwargs)
    else:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs
        )


def create_conv2d(
    in_channels: int, out_channels: int, kernel_size: Size2T, **kwargs
):
    """Select a 2d convolution implementation based on arguments. Creates and
    returns one of `torch.nn.Conv2d`, `Conv2dSame`, `MixedConv2d`, or
    `CondConv2d`. Used extensively by EfficientNet, MobileNetv3 and related
    networks.
    """
    if isinstance(kernel_size, list):
        # MixNet + CondConv combo not supported currently
        if "num_experts" in kwargs:
            raise ValueError
        # MixedConv groups are defined by kernel list
        if "groups" in kwargs:
            raise ValueError
        # We're going to use only lists for defining the MixedConv2d kernel
        # groups, ints, tuples, other iterables will continue to pass to
        # normal conv and specify h, w.
        m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop("depthwise", False)
        # for DW out_channels must be multiple of in_channels as must have
        # out_channels % groups == 0
        groups = in_channels if depthwise else kwargs.pop("groups", 1)
        if "num_experts" in kwargs and kwargs["num_experts"] > 0:
            m = CondConv2d(
				in_channels, out_channels, kernel_size, groups=groups,
				**kwargs
			)
        else:
            m = create_conv2d_pad(
				in_channels, out_channels, kernel_size, groups=groups, **kwargs
			)
    return m
