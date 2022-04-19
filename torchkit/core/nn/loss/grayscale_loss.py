#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Grayscale Loss.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from torchkit.core.factory import LOSSES
from torchkit.core.nn.loss.utils import weighted_sum
from torchkit.core.type import Tensors
from torchkit.core.type import Weights
from .mse_loss import elementwise_mse_loss

__all__ = [
    "grayscale_loss",
    "GrayscaleLoss",
]


# MARK: - GrayscaleLoss

def grayscale_loss(
    input             : Tensors,
    target            : Tensor,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Weights] = None,
    reduction         : str               = "mean",
) -> Tensor:
    """
    
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

    losses = []
    for inp in input:
        input_g  = torch.mean(inp,    1, keepdim=True)
        target_g = torch.mean(target, 1, keepdim=True)
        losses.append(
            elementwise_mse_loss(
                input_g, target_g, weight=elementwise_weight,
                reduction=reduction
            )
        )

    return weighted_sum(losses, input_weight)


@LOSSES.register(name="grayscale_loss", force=True)
class GrayscaleLoss(_Loss):
    """

    Attributes:
        name (str):
            Name of the loss. Default: `grayscale_loss`.
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
        self.name        = "grayscale_loss"
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
        return self.loss_weight * grayscale_loss(
            input, target,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
        )
