#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Color Constancy Loss.

A color constancy loss to correct the potential color deviations in the enhanced
image and also build the relations among the three adjusted channels.

References:
    https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from torchkit.core.factory import LOSSES
from torchkit.core.type import Tensors
from torchkit.core.type import Weights
from .utils import weighted_loss
from .utils import weighted_sum

__all__ = [
    "elementwise_color_constancy_loss",
    "color_constancy_loss",
    "ColorConstancyLoss",
]


# MARK: - ColorConstancyLoss

@weighted_loss
def elementwise_color_constancy_loss(
    input: Tensor, target: Optional[Tensor] = None
) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    x          = input
    mean_rgb   = torch.mean(x, [2, 3], keepdim=True)
    mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
    d_rg       = torch.pow(mr - mg, 2)
    d_rb       = torch.pow(mr - mb, 2)
    d_gb       = torch.pow(mb - mg, 2)
    k = torch.pow(
        torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2), 0.5
    )
    return k
    

def color_constancy_loss(
    input             : Tensors,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Weights] = None,
    reduction         : str               = "mean",
) -> Tensor:
    """Function that takes the mean element-wise absolute value difference.
    
    Args:
        input (Tensors):
            Prediction image of shape [B, C, H, W].
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
        elementwise_color_constancy_loss(
            input=i, target=None, weight=elementwise_weight, reduction=reduction
        ) for i in input
    ]
    return weighted_sum(losses, input_weight)
   
   
@LOSSES.register(name="color_constancy_loss")
class ColorConstancyLoss(_Loss):
    """Exposure Control Loss.

    Attributes:
        name (str):
            Name of the loss. Default: `color_constancy_loss`.
        loss_weight (Weights, optional):
			Weight for each loss value. Default: `1.0`.
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
        self, loss_weight: Optional[Weights] = 1.0, reduction: str = "mean"
    ):
        super().__init__(reduction=reduction)
        self.name        = "color_constancy_loss"
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
        return self.loss_weight * color_constancy_loss(
            input,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
        )
