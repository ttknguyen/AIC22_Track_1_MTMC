#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Smooth MAE loss.
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from torchkit.core.factory import LOSSES
from torchkit.core.type import Tensors
from torchkit.core.type import Weights
from .utils import weighted_loss
from .utils import weighted_sum

__all__ = [
    "elementwise_smooth_l1_loss",
    "elementwise_smooth_mae_loss",
    "smooth_l1_loss",
    "SmoothL1Loss",
    "smooth_mae_loss",
    "SmoothMAELoss",
]


# MARK: - SmoothMAELoss

@weighted_loss
def elementwise_smooth_mae_loss(
    input: Tensor, target: Tensor, beta: float = 1.0,
) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    return F.smooth_l1_loss(input, target, beta=beta, reduction="none")
    

def smooth_mae_loss(
    input             : Tensors,
    target            : Tensor,
    beta              : float             = 1.0,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Weights] = None,
    reduction         : str               = "mean",
) -> Tensor:
    """Function that takes the mean element-wise absolute value difference.
    
    Args:
        input (Tensors):
            Prediction image of shape [B, C, H, W].
        target (Tensor):
            Ground-truth image of shape [B, C, H, W].
        beta (float):
        
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
        elementwise_smooth_mae_loss(
            input=i, target=target, beta=beta, weight=elementwise_weight,
            reduction=reduction
        ) for i in input
    ]
    return weighted_sum(losses, input_weight)
   
   
@LOSSES.register(name="smooth_mae_loss")
class SmoothMAELoss(_Loss):
    """
    
    Attributes:
        name (str):
            Name of the loss. Default: `smooth_mae_loss`.
        beta (float):
            Default: `1.0`.
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
        beta       : float             = 1.0,
        loss_weight: Optional[Weights] = 1.0,
        reduction  : str               = "mean",
    ):
        super().__init__(reduction=reduction)
        self.name        = "smooth_mae_loss"
        self.beta        = beta
        self.loss_weight = loss_weight
        
        if self.reduction not in self.reductions:
            raise ValueError(f"Supported reduction are: {self.reductions}. "
                             f"But got: {self.reduction}.")
 
    # MARK: Forward Pass
    
    def forward(
        self,
        input             : Tensors,
        target            : Tensor,
        input_weight      : Optional[Weights] = None,
        elementwise_weight: Optional[Weights] = None,
        **_
    ) -> Tensor:
        return self.loss_weight * smooth_mae_loss(
            input, target, self.beta,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
        )


# MARK: - SmoothL1Loss

elementwise_smooth_l1_loss = elementwise_smooth_mae_loss
smooth_l1_loss             = smooth_mae_loss
SmoothL1Loss               = SmoothMAELoss

LOSSES.register(name="smooth_l1_loss", module=SmoothL1Loss)
