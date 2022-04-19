#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common Activation Layers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchkit.core.factory import ACT_LAYERS
from torchkit.core.type import Callable
from torchkit.core.type import Size2T
from torchkit.core.type import to_2tuple

__all__ = [
    "ArgMax", 
    "Clamp", 
    "Clip", 
    "create_act_layer", 
    "FReLU", 
    "GELU",
    "gelu",
    "hard_mish",
    "hard_swish_yolov4",
    "HardMish",
    "HardSwishYoloV4",
    "MemoryEfficientMish",
    "MemoryEfficientSwish",
    "Mish",
    "mish", 
    "PReLU",
    "Sigmoid",
    "sigmoid",
    "Swish",
    "swish",
    "Tanh",
    "tanh"
]


# MARK: - Register

ACT_LAYERS.register(name="celu",         module=nn.CELU)
ACT_LAYERS.register(name="elu",          module=nn.ELU)
ACT_LAYERS.register(name="gelu",         module=nn.GELU)
ACT_LAYERS.register(name="hard_shrink",  module=nn.Hardshrink)
ACT_LAYERS.register(name="hard_sigmoid", module=nn.Hardsigmoid)
ACT_LAYERS.register(name="hard_swish", 	 module=nn.Hardswish)
ACT_LAYERS.register(name="hard_tanh",    module=nn.Hardtanh)
ACT_LAYERS.register(name="identity",     module=nn.Identity)
ACT_LAYERS.register(name="leaky_relu",   module=nn.LeakyReLU)
ACT_LAYERS.register(name="log_sigmoid",  module=nn.LogSigmoid)
ACT_LAYERS.register(name="log_softmax",  module=nn.LogSoftmax)
ACT_LAYERS.register(name="prelu",        module=nn.PReLU)
ACT_LAYERS.register(name="relu", 		 module=nn.ReLU)
ACT_LAYERS.register(name="relu6", 		 module=nn.ReLU6)
ACT_LAYERS.register(name="rrelu", 		 module=nn.RReLU)
ACT_LAYERS.register(name="selu", 		 module=nn.SELU)
ACT_LAYERS.register(name="sigmoid",		 module=nn.Sigmoid)
ACT_LAYERS.register(name="silu", 		 module=nn.SiLU)
ACT_LAYERS.register(name="softmax",      module=nn.Softmax)
ACT_LAYERS.register(name="softmin",      module=nn.Softmin)
ACT_LAYERS.register(name="softplus", 	 module=nn.Softplus)
ACT_LAYERS.register(name="softshrink",   module=nn.Softshrink)
ACT_LAYERS.register(name="softsign",     module=nn.Softsign)
ACT_LAYERS.register(name="tanh",		 module=nn.Tanh)
ACT_LAYERS.register(name="tanhshrink",   module=nn.Tanhshrink)


# MARK: - ArgMax

@ACT_LAYERS.register(name="arg_max")
class ArgMax(nn.Module):
    """Find the indices of the maximum value of all elements in the input
    image.
    
    Attributes:
        dim (int, optional):
            Dimension to find the indices of the maximum value.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        return torch.argmax(input, dim=self.dim)


# MARK: - Clamp/Clip

@ACT_LAYERS.register(name="clamp")
class Clamp(nn.Module):
    """Clamp activation layer. This activation function is to clamp the feature
    map value within :math:`[min, max]`. More details can be found in
    `torch.clamp()`.
    
    Attributes:
        min (float):
            Lower-bound of the range to be clamped to.
        max (float):
            Upper-bound of the range to be clamped to.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, min: float = -1.0, max: float = 1.0):
        super().__init__()
        self.min = min
        self.max = max
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        return torch.clamp(input, min=self.min, max=self.max)


Clip = Clamp
ACT_LAYERS.register(name="clip", module=Clip)


# MARK: - FReLU

@ACT_LAYERS.register(name="frelu")
class FReLU(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(self, c1: int, k: Size2T = 3):
        super().__init__()
        k         = to_2tuple(k)
        self.conv = nn.Conv2d(c1, c1, k, (1, 1), 1, groups=c1)
        self.bn   = nn.BatchNorm2d(c1)
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        return torch.max(input, self.bn(self.conv(input)))


# MARK: - GELU

def gelu(input: Tensor, inplace: bool = False) -> Tensor:
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    return F.gelu(input)


@ACT_LAYERS.register(name="gelu", force=True)
class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg).
    """

    # MARK: Magic Functions

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        return gelu(input, self.inplace)


# MARK: - HardMish

def hard_mish(input: Tensor, inplace: bool = False) -> Tensor:
    """Hard Mish Experimental, based on notes by Mish author Diganta Misra at
    https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    if inplace:
        return input.mul_(0.5 * (input + 2).clamp(min=0, max=2))
    else:
        return 0.5 * input * (input + 2).clamp(min=0, max=2)


@ACT_LAYERS.register(name="hard_mish")
class HardMish(nn.Module):

    # MARK: Magic Functions

    def __init__(self, inplace: bool = False):
        super(HardMish, self).__init__()
        self.inplace = inplace

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        return hard_mish(input, self.inplace)
      

# MARK: - HardSwishYoloV4

def hard_swish_yolov4(input: Tensor, inplace: bool = False) -> Tensor:
    return input * F.hardtanh(input + 3, 0.0, 6.0, inplace) / 6.0


@ACT_LAYERS.register(name="hard_swish_yolov4")
class HardSwishYoloV4(nn.Module):

    # MARK: Magic Functions

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        return hard_swish_yolov4(input, self.inplace)
    

# MARK: - MemoryEfficientMish

@ACT_LAYERS.register(name="memory_efficient_mish")
class MemoryEfficientMish(nn.Module):
    
    # noinspection PyMethodOverriding
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input.mul(torch.tanh(F.softplus(input)))  # input * tanh(ln(1 + exp(input)))

        @staticmethod
        def backward(ctx, grad_output):
            input = ctx.saved_tensors[0]
            sx = torch.sigmoid(input)
            fx = F.softplus(input).tanh()
            return grad_output * (fx + input * sx * (1 - fx * fx))

    def forward(self, input: Tensor) -> Tensor:
        return self.F.apply(input)
    

# MARK: - MemoryEfficientSwish

@ACT_LAYERS.register(name="memory_efficient_swish")
class MemoryEfficientSwish(nn.Module):
    
    # noinspection PyMethodOverriding
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input * torch.sigmoid(input)

        @staticmethod
        def backward(ctx, grad_output):
            input  = ctx.saved_tensors[0]
            sx = torch.sigmoid(input)
            return grad_output * (sx * (1 + input * (1 - sx)))

    def forward(self, input: Tensor) -> Tensor:
        return self.F.apply(input)


# MARK: - Mish

def mish(input: Tensor, inplace: bool = False) -> Tensor:
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function -
    https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return input.mul(F.softplus(input).tanh())


@ACT_LAYERS.register(name="mish")
class Mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function -
    https://arxiv.org/abs/1908.08681
    """

    # MARK: Magic Functions

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    # MARK: Forward Pass

    def forward(self, input: Tensor) -> Tensor:
        return mish(input)


# MARK: - PReLU

@ACT_LAYERS.register(name="prelu", force=True)
class PReLU(nn.PReLU):
    """Applies PReLU (w/ dummy inplace arg)."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        num_parameters: int   = 1,
        init          : float = 0.25,
        inplace       : bool  = False
    ):
        super().__init__(num_parameters=num_parameters, init=init)
        self.inplace = inplace
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        return F.prelu(input, self.weight)


# MARK: - Sigmoid

def sigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    return input.sigmoid_() if inplace else input.sigmoid()


@ACT_LAYERS.register(name="sigmoid", force=True)
class Sigmoid(nn.Module):
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        return input.sigmoid_() if self.inplace else input.sigmoid()


# MARK: - Swish

def swish(input: Tensor, inplace: bool = False) -> Tensor:
    """Swish described in: https://arxiv.org/abs/1710.05941"""
    return input.mul_(input.sigmoid()) if inplace else input.mul(input.sigmoid())


@ACT_LAYERS.register(name="swish")
class Swish(nn.Module):
    """Swish Module. This module applies the swish function."""
    
    # MARK: Magic Functions
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        return swish(input, self.inplace)


# MARK: - Tanh

def tanh(input: Tensor, inplace: bool = False) -> Tensor:
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    return input.tanh_() if inplace else input.tanh()


@ACT_LAYERS.register(name="tanh", force=True)
class Tanh(nn.Module):
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        return input.tanh_() if self.inplace else input.tanh()


# MARK: - Builder

def create_act_layer(
    apply_act: bool               = True,
    act_layer: Optional[Callable] = nn.ReLU,
    inplace  : bool               = True,
    **_
) -> nn.Module:
    """Create activation layer."""
    if isinstance(act_layer, str):
        act_layer = ACT_LAYERS.build(name=act_layer)
    if (act_layer is not None) and apply_act:
        act_args  = dict(inplace=True) if inplace else {}
        act_layer = act_layer(**act_args)
    else:
        act_layer = nn.Identity()
    return act_layer
