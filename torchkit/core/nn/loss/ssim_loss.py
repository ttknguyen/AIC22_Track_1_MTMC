#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create a criterion that computes a loss based on the SSIM measurement.

https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/ssim.html
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from torchkit.core.factory import LOSSES
from torchkit.core.nn.metric import ssim
from torchkit.core.type import Tensors
from torchkit.core.type import Weights
from .utils import weighted_loss
from .utils import weighted_sum

__all__ = [
    "elementwise_ssim_loss",
    "ssim_loss",
    "SSIMLoss",
]


# MARK: - SSIMLoss

@weighted_loss
def elementwise_ssim_loss(
    input      : Tensor,
    target     : Tensor,
    window_size: int,
    max_val    : float = 1.0,
    eps        : float = 1e-12
) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    # Compute the ssim map
    ssim_map = ssim(input, target, window_size, max_val, eps)
    # Compute and reduce the loss
    return torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)


def ssim_loss(
    input             : Tensors,
    target            : Tensor,
    window_size       : int,
    max_val           : float             = 1.0,
    eps               : float             = 1e-12,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Weights] = None,
    reduction         : str               = "mean"
) -> Tensor:
    """Function that computes a loss based on the SSIM measurement:

    .. math::
      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    Args:
        input (Tensors):
            Prediction image of shape [B, C, H, W].
        target (Tensor):
            Ground-truth image of shape [B, C, H, W].
        window_size (int):
            Size of the gaussian kernel to smooth the images.
        max_val (float):
            Dynamic range of the images.
        eps (float):
            Small value for numerically stability when dividing.
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
            Loss based on the ssim index.
    """
    if isinstance(input, Tensor):  # Single output
        input = [input]
    elif isinstance(input, dict):
        input = list(input.values())
    if not isinstance(input, (list, tuple)):
        raise ValueError(f"Do not support input of type: {type(input)}.")

    losses = [
        elementwise_ssim_loss(
            input=i, target=target, window_size=window_size, max_val=max_val,
            eps=eps, weight=elementwise_weight, reduction=reduction,
        ) for i in input
    ]
    return weighted_sum(losses, input_weight)
  
  
@LOSSES.register(name="ssim_loss")
class SSIMLoss(_Loss):
    """Create a criterion that computes a loss based on the SSIM measurement.
    Supports both single output and multi-outputs input.

    Attributes:
        name (str):
            Name of the loss. Default: `ssim_loss`.
        max_val (float):
            Dynamic range of the images.
        eps (float):
            Small value for numerically stability when dividing.
        loss_weight (Weights, optional):
			Weight for each loss value. Default: `1.0`.
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

    reductions = ["none", "mean", "sum"]

    # MARK: Magic Functions
    
    def __init__(
        self,
        window_size: int,
        max_val    : float             = 1.0,
        eps        : float             = 1e-12,
        loss_weight: Optional[Weights] = 1.0,
        reduction  : str               = "mean",
    ):
        super().__init__(reduction=reduction)
        self.name        = "ssim_loss"
        self.window_size = window_size
        self.max_val     = max_val
        self.eps         = eps
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
        return self.loss_weight * ssim_loss(
            input, target,
            window_size        = self.window_size,
            max_val            = self.max_val,
            eps                = self.eps,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
        )
