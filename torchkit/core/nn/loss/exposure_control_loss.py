#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Exposure Control Loss.

The exposure control loss measures the distance between the average intensity
value of a local region to the well-exposedness level E.

References:
    https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from torchkit.core.factory import LOSSES
from torchkit.core.type import Size2T
from torchkit.core.type import Tensors
from torchkit.core.type import Weights
from .utils import weighted_loss
from .utils import weighted_sum

__all__ = [
    "elementwise_exposure_control_loss",
    "exposure_control_loss",
    "ExposureControlLoss",
]

pool = nn.AvgPool2d(16)


# MARK: - ExposureControlLoss

@weighted_loss
def elementwise_exposure_control_loss(
    input     : Tensor,
    target    : Optional[Tensor],
    patch_size: Size2T,
    mean_val  : float
) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    global pool
    if pool.kernel_size != patch_size:
        pool = nn.AvgPool2d(patch_size)
    
    x    = input
    x    = torch.mean(x, 1, keepdim=True)
    mean = pool(x)
    d    = torch.pow(mean - torch.FloatTensor([mean_val]).to(x.device), 2)
    return d
    

def exposure_control_loss(
    input             : Tensors,
    patch_size        : Size2T,
    mean_val          : float,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Tensor]  = None,
    reduction         : str               = "mean",
) -> Tensor:
    """Function that takes the mean element-wise absolute value difference.
    
    Args:
        input (Tensors):
            Prediction image of shape [B, C, H, W].
        patch_size (Size2T):
            Kernel size for pooling layer.
        mean_val (float):
        
        input_weight (Weights, optional):
            Weight for each input of shape [N]. Default: `None`.
        elementwise_weight (Weights, optional):
            Element-wise weights of shape [B, C, H, W]. Default: `None`.
        reduction (str):
            Specifies the reduction to apply to the output.
            One of: [`none`, `mean`, `sum`].
            - `none`: No reduction will be applied.
            - `mean`: The sum of the output will be divided by the number of
                      elements in the output.
            - `sum`: The output will be summed.
            Default: `mean`.
    """
    if isinstance(input, Tensor):  # Single output
        input = [input]
    elif isinstance(input, dict):
        input = list(input.values())
    if not isinstance(input, (list, tuple)):
        raise ValueError(f"Do not support input of type: {type(input)}.")
    
    losses = [
        elementwise_exposure_control_loss(
            input=i, target=None, patch_size=patch_size, mean_val=mean_val,
            weight=elementwise_weight, reduction=reduction
        ) for i in input
    ]
    return weighted_sum(losses, input_weight)
   
   
@LOSSES.register(name="exposure_control_loss")
class ExposureControlLoss(_Loss):
    """Exposure Control Loss.

    Attributes:
        name (str):
            Name of the loss. Default: `exposure_control_loss`.
        patch_size (Size2T):
            Kernel size for pooling layer.
        mean_val (float):
        
        loss_weight (Weights, optional):
			Weighted values for each element in `input`.
        reduction (str):
            Specifies the reduction to apply to the output.
            One of: [`none`, `mean`, `sum`].
            - `none`: No reduction will be applied.
            - `mean`: The sum of the output will be divided by the number of
                      elements in the output.
            - `sum`: The output will be summed.
            Default: `mean`.
    """
    
    reductions = ["none", "mean", "sum"]
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        patch_size : Size2T,
        mean_val   : float,
        loss_weight: Optional[Weights] = 1.0,
        reduction  : str               = "mean"
    ):
        super().__init__(reduction=reduction)
        self.name        = "exposure_control_loss"
        self.patch_size  = patch_size
        self.mean_val    = mean_val
        self.loss_weight = loss_weight
        
        if self.reduction not in self.reductions:
            raise ValueError(f"Supported reduction are: {self.reductions}. "
                             f"But got: {self.reduction}.")
 
    # MARK: Forward Pass
    
    def forward(
        self,
        input             : Tensors,
        input_weight      : Optional[Weights] = None,
        elementwise_weight: Optional[Weights] = None,
        **_
    ) -> Tensor:
        return self.loss_weight * exposure_control_loss(
            input              = input,
            patch_size         = self.patch_size,
            mean_val           = self.mean_val,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
        )
