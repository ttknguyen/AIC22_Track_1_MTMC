#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""PSNR loss.
Modified from BasicSR: https://github.com/xinntao/BasicSR
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from torchkit.core.factory import LOSSES
from torchkit.core.nn.metric import psnr
from torchkit.core.type import Tensors
from torchkit.core.type import Weights
from .utils import weighted_loss
from .utils import weighted_sum

__all__ = [
    "elementwise_psnr_loss",
    "psnr_loss",
    "PSNRLoss",
]


# MARK: - PSNRLoss

@weighted_loss
def elementwise_psnr_loss(
    input: Tensor, target: Tensor, max_val: float = 1.0
) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    return -1.0 * psnr(input, target, max_val)


def psnr_loss(
    input             : Tensors,
    target            : Tensor,
    max_val           : float             = 1.0,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Weights] = None,
    reduction         : str               = "mean",
) -> Tensor:
    """Function that computes a loss based on the PSNR measurement:

    .. math::
      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    Args:
        input (Tensors):
            Prediction image of shape [B, C, H, W].
        target (Tensor):
            Ground-truth image of shape [B, C, H, W].
        max_val (float):
            Dynamic range of the images.
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
    
    Returns:
        (Tensor):
            Loss based on the SSIM index.
            
    Examples:
        >>> ones = torch.ones(1)
        >>> psnr_loss(ones, 1.2 * ones, 2.)
        >>> # 10 * log(4/((1.2-1)**2)) / log(10)
        >>> # image(-20.0000)
    """
    if isinstance(input, Tensor):  # Single output
        input = [input]
    elif isinstance(input, dict):
        input = list(input.values())
    if not isinstance(input, (list, tuple)):
        raise ValueError(f"Do not support input of type: {type(input)}.")

    losses = [
        elementwise_psnr_loss(
            input=i, target=target, max_val=max_val, weight=elementwise_weight,
            reduction=reduction
        ) for i in input
    ]
    return weighted_sum(losses, input_weight)
   
   
@LOSSES.register(name="psnr_loss")
class PSNRLoss(_Loss):
    """PSNR Loss. Supports both single output and multi-outputs input.
    
    Attributes:
        name (str):
            Name of the loss. Default: `psnr_loss`.
        max_val (float):
            Dynamic range of the images.
        loss_weight (Weights, optional):
			Weighted values for each element in `input`.
        reduction (str):
            reduction (str):
            Specifies the reduction to apply to the output.
            One of: [`none`, `mean`, `sum`].
            - `none`: No reduction will be applied.
            - `mean`: The sum of the output will be divided by the number of
                      elements in the output.
            - `sum`: The output will be summed.
            Default: `mean`.
    """

    reductions = ["mean"]
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        max_val    : float             = 1.0,
        loss_weight: Optional[Weights] = 1.0,
        reduction  : str               = "mean"
    ):
        super().__init__(reduction=reduction)
        self.name        = "psnr_loss"
        self.max_val     = max_val
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
        return self.loss_weight * psnr_loss(
            input, target, self.max_val,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
        )
