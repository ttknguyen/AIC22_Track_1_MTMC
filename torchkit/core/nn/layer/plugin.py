#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from typing import Sequence
from typing import Union

import torch
from torch import nn as nn
from torch import Tensor

from torchkit.core.factory import PLUGIN_LAYERS
from torchkit.core.type import ListOrTupleAnyT
from torchkit.core.type import Size2T

__all__ = [
    "Concat", 
    "ContextBlock", 
    "Flatten",
    "Focus", 
    "Mean", 
    "Scale",
    "Sum"
]


# MARK: - ContextBlock

@PLUGIN_LAYERS.register(name="context_block")
class ContextBlock(nn.Module):
    """ContextBlock module in GCNet. See 'GCNet: Non-local Networks Meet
    Squeeze-Excitation Networks and Beyond'
    (https://arxiv.org/abs/1904.11492) for details.
    
    Args:
        in_channels (int):
            Channels of the input feature map.
        ratio (float):
            Ratio of channels of transform bottleneck
        pooling_type (str):
            Pooling method for context modeling. One of: [`att`, `avg`].
            `att` stands for attention pooling and `avg` stands for average
            pooling. Default: `att`.
        fusion_types (Sequence[str]):
            Fusion method for feature fusion, One of: [`channels_add`,
            `channel_mul`]. `channels_add` stands for channelwise addition
            and `channel_mul` stands for multiplication.
            Default: (`channel_add`,).
        
    """
    
    _abbr_ = "context_block"
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        ratio       : float,
        pooling_type: str           = "att",
        fusion_types: Sequence[str] = ("channel_add", ),
        *args, **kwargs
    ):
        super().__init__()
        
        if pooling_type not in ["avg", "att"]:
            raise ValueError
        if not isinstance(fusion_types, (list, tuple)):
            raise ValueError
        
        valid_fusion_types = ["channel_add", "channel_mul"]
        if not all([f in valid_fusion_types for f in fusion_types]):
            raise ValueError
        if len(fusion_types) <= 0:
            raise ValueError("At least one fusion should be used.")
        
        planes = int(in_channels * ratio)
        
        if pooling_type == "att":
            self.conv_mask = nn.Conv2d(
                in_channels=in_channels, out_channels=1, kernel_size=(1, 1)
            )
            self.softmax   = nn.Softmax(dim=2)
        else:
            self.avg_pool  = nn.AdaptiveAvgPool2d(1)
        
        if "channel_add" in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=planes,
                    kernel_size=(1, 1)
                ),
                nn.LayerNorm(normalized_shape=[planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=planes, out_channels=in_channels,
                    kernel_size=(1, 1)
                )
            )
        else:
            self.channel_add_conv = None
        
        if "channel_mul" in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=planes,
                    kernel_size=(1, 1)
                ),
                nn.LayerNorm(normalized_shape=[planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=planes, out_channels=in_channels,
                    kernel_size=(1, 1)
                )
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        # [N, C, 1, 1]
        context = self.spatial_pool(input=input)
        pred    = input
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            pred            *= channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            pred             = pred + channel_add_term
        
        return pred
    
    def spatial_pool(self, input: Tensor) -> Tensor:
        batch, channel, height, width = input.size()
        if self.pooling_type == "att":
            input_x = input
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(input)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(input)
        
        return context


# MARK: - Concat

@PLUGIN_LAYERS.register(name="concat")
class Concat(nn.Module):
    """Concatenate a list of tensors along dimension.
    
    Args:
        dim (str, ellipsis, None):
            Dimension to concat to. Default: `1`.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, dim: Union[str, ellipsis, None] = 1):
        super().__init__()
        self.dim = dim
        
    # MARK: Forward Pass
    
    def forward(self, input: ListOrTupleAnyT[Tensor]) -> Tensor:
        """Run forward pass.

        Args:
            input (ListOrTupleAnyT[Tensor]):
                A list of tensors along dimension.

        Returns:
            pred (Tensor):
                Flattened image.
        """
        return torch.cat(input, dim=self.dim)


# MARK: - Flatten

# noinspection PyMethodMayBeStatic
@PLUGIN_LAYERS.register(name="flatten")
class Flatten(nn.Module):
    """Flatten the image. Commonly used after `nn.AdaptiveAvgPool2d(1)` to
    remove last 2 dimensions.
    
    Attributes:
        channels (int):
            Channels to flatten the features to. Default: `-1`.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, channels: int = -1):
        super().__init__()
        self.channels = channels
        
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        pred = input.view(input.shape[0], self.channels)
        return pred


# MARK: - Focus

@PLUGIN_LAYERS.register(name="focus")
class Focus(nn.Module):
    """Focus wh information into c-space.
    
    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (Size2T):
            Size of the convolving kernel. Default: `1`.
        stride (Size2T):
            Stride of the convolution. Default: `1`.
        padding (str, int, Size2T, optional):
            Zero-padding added to both sides of the input. Default: `None`.
        groups (int):
            Default: `1`.
        apply_act (bool):
            Should use activation layer. Default: `True`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T                   = 1,
        stride      : Size2T                   = 1,
        padding     : Union[int, Size2T, None] = None,
        groups      : int                      = 1,
        apply_act   : bool                     = True,
    ):
        super().__init__()
        from torchkit.core.nn.layer import ConvBnMish
        self.conv = ConvBnMish(
            in_channels  = in_channels * 4,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            groups       = groups,
            apply_act    = apply_act,
        )
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            input (Tensor):
                Input image.

        Returns:
            pred (Tensor):
                Output image. input(b,c,w,h) -> pred(b,4c,w/2,h/2)
        """
        return self.conv(
            torch.cat([input[..., ::2, ::2], input[..., 1::2, ::2],
                       input[..., ::2, 1::2], input[..., 1::2, 1::2]], 1)
        )


# MARK: - Mean

# noinspection PyMethodMayBeStatic
@PLUGIN_LAYERS.register(name="mean")
class Mean(nn.Module):
    """Calculate mean of the image.
    
    Attributes:
        dim:
            Default: `None`.
        keepdim (bool):
            Default: `False`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dim    : Sequence[Union[str, ellipsis, None]] = None,
        keepdim: bool = False,
    ):
        super().__init__()
        self.dim     = dim
        self.keepdim = keepdim
        
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        return input.mean(dim=self.dim, keepdim=self.keepdim)
    

# MARK: - Scale

@PLUGIN_LAYERS.register(name="scale")
class Scale(nn.Module):
    """A learnable scale parameter. This layer scales the input by a learnable
    factor. It multiplies a learnable scale parameter of shape (1,) with
    input of any shape.
    
    Attributes:
        scale (float):
            Initial value of scale factor. Default: `1.0`.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, scale: float = 1.0):
        """
        
        Args:
            scale (float):
                Initial value of scale factor. Default: `1.0`.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        return input * self.scale


# MARK: - Sum

@PLUGIN_LAYERS.register(name="sum")
class Sum(nn.Module):
    """Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070.
    
    Args:
    	n (int):
    		Number of inputs.
    """

    # MARK: Magic Functions

    def __init__(self, n: int, weight: bool = False):
        super().__init__()
        self.weight = weight  # Apply weights boolean
        self.iter 	= range(n - 1)  # iter object
        if weight:
            # Layer weights
            self.w = nn.Parameter(
				-torch.arange(1.0, n) / 2, requires_grad=True
			)

    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        y = input[0]
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + input[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + input[i + 1]
        return y
