#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Illumination Smoothness Loss.

To preserve the mono-tonicity relations between neighboring pixels, we add an
illumination smoothness loss to each curve parameter map A.

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
    "elementwise_illumination_smoothness_loss",
    "illumination_smoothness_loss",
    "IlluminationSmoothnessLoss",
]


# MARK: - IlluminationSmoothnessLoss

@weighted_loss
def elementwise_illumination_smoothness_loss(
    input: Tensor, target: Optional[Tensor], tv_loss_weight: int
) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    x          = input
    batch_size = x.size()[0]
    h_x        = x.size()[2]
    w_x        = x.size()[3]
    count_h    = (x.size()[2] - 1) * x.size()[3]
    count_w    = x.size()[2] * (x.size()[3] - 1)
    h_tv       = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
    w_tv       = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
    return tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    

def illumination_smoothness_loss(
    input             : Tensors,
    tv_loss_weight    : int,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Weights] = None,
    reduction         : str               = "mean",
) -> Tensor:
    """Function that takes the mean element-wise absolute value difference.
    
    Args:
        input (Tensors):
            Prediction image of shape [B, C, H, W].
        tv_loss_weight (int):
        
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
        elementwise_illumination_smoothness_loss(
            input=i, tv_loss_weight=tv_loss_weight, weight=elementwise_weight,
            reduction=reduction
        ) for i in input
    ]
    return weighted_sum(losses, input_weight)
   
   
@LOSSES.register(name="illumination_smoothness_loss")
class IlluminationSmoothnessLoss(_Loss):
    """Exposure Control Loss.

    Attributes:
        name (str):
            Name of the loss. Default: `illumination_smoothness_loss`.
        tv_loss_weight (int):
        
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
        self,
        tv_loss_weight: int               = 1,
        loss_weight   : Optional[Weights] = 1.0,
        reduction     : str               = "mean"
    ):
        super().__init__(reduction=reduction)
        self.name           = "illumination_smoothness_loss"
        self.tv_loss_weight = tv_loss_weight
        self.loss_weight    = loss_weight
        
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
        return self.loss_weight * illumination_smoothness_loss(
            input              = input,
            tv_loss_weight     = self.tv_loss_weight,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
        )
