#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Gradient Loss.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from torchkit.core.factory import LOSSES
from torchkit.core.nn.loss.utils import weighted_sum
from torchkit.core.type import Tensors
from torchkit.core.type import Weights

__all__ = [
    "GradientLoss",
]


# MARK: - GradientLoss

def gradient_loss(input: Tensors, input_weight: Optional[Weights] = None) -> Tensor:
    """
    
    Args:
        input (Tensors):
            Prediction image of shape [B, C, H, W].
        input_weight (Weights, optional):
            Weight for each input of shape [N]. Default: `None`.
    """
    if isinstance(input, Tensor):  # Single output
        input = [input]
    elif isinstance(input, dict):
        input = list(input.values())
    if not isinstance(input, (list, tuple)):
        raise ValueError(f"Do not support input of type: {type(input)}.")

    losses = []
    for inp in input:
        gradient_a_x = torch.abs(inp[:, :, :, :-1] - inp[:, :, :, 1:])
        gradient_a_y = torch.abs(inp[:, :, :-1, :] - inp[:, :, 1:, :])
        losses.append(torch.mean(gradient_a_x) + torch.mean(gradient_a_y))

    return weighted_sum(losses, input_weight)


@LOSSES.register(name="gradient_loss", force=True)
class GradientLoss(_Loss):
    """L1 loss on the gradient of the image.

    Attributes:
        name (str):
            Name of the loss. Default: `gradient_loss`.
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
        self.name        = "gradient_loss"
        self.loss_weight = loss_weight

        if self.reduction not in self.reductions:
            raise ValueError(f"Supported reduction are: {self.reductions}. "
                             f"But got: {self.reduction}.")
        
    # MARK: Forward Pass
    
    def forward(
        self,
        input       : Tensors,
        input_weight: Optional[Weights] = None,
        **_
    ) -> Tensor:
        return self.loss_weight * gradient_loss(input, input_weight)
