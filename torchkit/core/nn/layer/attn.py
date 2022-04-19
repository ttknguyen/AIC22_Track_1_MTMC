#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention Layers.
"""

from __future__ import annotations

import torch
from torch import nn
from torch import Tensor

from torchkit.core.factory import ATTN_LAYERS
from torchkit.core.type import Size2T
from torchkit.core.type import to_2tuple

__all__ = [
    "CAB",
	"CAL",
	"ChannelAttentionBlock",
	"ChannelAttentionLayer",
	"SAM",
	"SupervisedAttentionModule"
]


# MARK: - Register

ATTN_LAYERS.register(name="identity", module=nn.Identity)


# MARK: - ChannelAttentionLayer

@ATTN_LAYERS.register(name="channel_attention_layer")
class ChannelAttentionLayer(nn.Module):
	"""Channel Attention Layer.
	
	Attributes:
		channels (int):
			Number of input and output channels.
		reduction (int):
			Reduction factor. Default: `16`.
		bias (bool):
			Default: `False`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(self, channels: int, reduction: int = 16, bias: bool = False):
		super().__init__()
		# Global average pooling: feature --> point
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		# Feature channel downscale and upscale --> channel weight
		self.conv_du  = nn.Sequential(
			nn.Conv2d(channels, channels // reduction, kernel_size=(1, 1),
					  padding=0, bias=bias),
			nn.ReLU(inplace=True),
			nn.Conv2d(channels // reduction, channels, kernel_size=(1, 1),
					  padding=0, bias=bias),
			nn.Sigmoid()
		)
	
	# MARK: Forward Pass
	
	def forward(self, input: Tensor) -> Tensor:
		out = self.avg_pool(input)
		out = self.conv_du(out)
		out = input * out
		return out


CAL = ChannelAttentionLayer
ATTN_LAYERS.register(name="cal", module=CAL)


# MARK: - ChannelAttentionBlock

@ATTN_LAYERS.register(name="channel_attention_block")
class ChannelAttentionBlock(nn.Module):
	"""Channel Attention Block.
	
	Attributes:
		channels (int):
			Number of input and output channels.
		kernel_size (Size2T):
			Kernel size of the convolution layer.
		reduction (int):
			Reduction factor.
		bias (bool):
			Default: `False`.
		act (nn.Module):
			Activation layer.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		channels   : int,
		kernel_size: Size2T,
		reduction  : int,
		bias       : bool,
		act        : nn.Module,
	):
		super().__init__()
		kernel_size = to_2tuple(kernel_size)
		padding     = kernel_size[0] // 2
		stride		= (1, 1)
		self.ca     = CAL(channels, reduction, bias)
		self.body   = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size, stride, padding,
					  bias=bias),
			act,
			nn.Conv2d(channels, channels, kernel_size, stride, padding,
					  bias=bias),
		)
		
	# MARK: Forward Pass
	
	def forward(self, input: Tensor) -> Tensor:
		pred = self.body(input)
		pred = self.ca(pred)
		pred += input
		return pred


CAB = ChannelAttentionBlock
ATTN_LAYERS.register(name="cab")


# MARK: - SupervisedAttentionModule

@ATTN_LAYERS.register(name="supervised_attention_module")
class SupervisedAttentionModule(nn.Module):
	"""Supervised Attention Module.
	
	Args:
		channels (int):
			Number of input channels.
		kernel_size (Size2T):
			Kernel size of the convolution layer.
		bias (bool):
			Default: `False`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(self, channels: int, kernel_size: Size2T, bias: bool):
		super().__init__()
		kernel_size = to_2tuple(kernel_size)
		padding     = kernel_size[0] // 2
		stride		= (1, 1)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride,
							   padding, bias=bias)
		self.conv2 = nn.Conv2d(channels, 3,        kernel_size, stride,
							   padding, bias=bias)
		self.conv3 = nn.Conv2d(3,        channels, kernel_size, stride,
							   padding, bias=bias)
		
	# MARK: Forward Pass
	
	def forward(self, fx: Tensor, input: Tensor) -> tuple[Tensor, Tensor]:
		"""Run forward pass.

		Args:
			fx (Tensor):
				Output from previous steps.
			input (Tensor):
				Original input images.
			
		Returns:
			pred (Tensor):
				Output image.
			img (Tensor):
				Output image.
		"""
		x1    = self.conv1(fx)
		img   = self.conv2(fx) + input
		x2    = torch.sigmoid(self.conv3(img))
		x1   *= x2
		x1   += fx
		pred  = x1
		return pred, img


SAM = SupervisedAttentionModule
ATTN_LAYERS.register(name="sam")
