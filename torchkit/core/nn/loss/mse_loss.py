#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MSE loss.
Modified based on BasicSR: https://github.com/xinntao/BasicSR
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
    "elementwise_l2_loss",
    "elementwise_mse_loss",
    "l2_loss",
    "L2Loss",
    "mse_loss",
    "MSELoss",
]


# MARK: - MSELoss

@weighted_loss
def elementwise_mse_loss(input: Tensor, target: Tensor) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    return F.mse_loss(input, target, reduction="none")


def mse_loss(
    input             : Tensors,
    target            : Tensor,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Weights] = None,
    reduction         : str               = "mean",
) -> Tensor:
    """Measures the element-wise mean squared error.

    Args:
        input (Tensors):
            Prediction image of shape [B, C, H, W].
        target (Tensor):
            Ground-truth image of shape [B, C, H, W].
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
        elementwise_mse_loss(
            input=i, target=target, weight=elementwise_weight,
            reduction=reduction
        ) for i in input
    ]
    return weighted_sum(losses, input_weight)
   
    
@LOSSES.register(name="mse_loss", force=True)
class MSELoss(_Loss):
    """MSE (Mean Squared Error or L2) loss. Supports both single input and
    multi-input with weighted sum.

    Attributes:
        name (str):
            Name of the loss. Default: `mse_loss`.
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
        self.name        = "mse_loss"
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
        return self.loss_weight * mse_loss(
            input, target,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
        )


# MARK: - L2Loss

elementwise_l2_loss = elementwise_mse_loss
l2_loss             = mse_loss
L2Loss              = MSELoss

LOSSES.register(name="l2_loss", module=L2Loss)
