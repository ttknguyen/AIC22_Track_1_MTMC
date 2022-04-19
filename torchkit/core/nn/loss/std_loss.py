#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Std Loss.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional
from torch.nn.modules.loss import _Loss

from torchkit.core import LOSSES
from torchkit.core.type import Tensors
from torchkit.core.type import Weights
from .mse_loss import elementwise_mse_loss
from .utils import weighted_sum

__all__ = [
    "std_loss",
    "StdLoss",
]


blur        = (1 / 25) * np.ones((5, 5))
blur        = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
blur        = nn.Parameter(data=torch.FloatTensor(blur), requires_grad=False)

image       = np.zeros((5, 5))
image[2, 2] = 1
# noinspection PyArgumentList
image       = image.reshape(1, 1, image.shape[0], image.shape[1])
image       = nn.Parameter(data=torch.FloatTensor(image), requires_grad=False)


# MARK: - StdLoss


def std_loss(
    input             : Tensors,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Weights] = None,
    reduction         : str               = "mean",
) -> Tensor:
    """
    
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
    
    global blur, image
    if blur.device != input[0].device:
        blur = blur.to(input[0].device)
    if image.device != input[0].device:
        image = image.to(input[0].device)
        
    losses = []
    for inp in input:
        inp = torch.mean(inp, 1, keepdim=True)
        losses.append(
            elementwise_mse_loss(
                functional.conv2d(inp, image),
                functional.conv2d(inp, blur),
                weight    = elementwise_weight,
                reduction = reduction
            )
        )
        
    return weighted_sum(losses, input_weight)


@LOSSES.register(name="std_loss")
class StdLoss(_Loss):
    """Loss on the variance of the image. Works in the grayscale.
    If the image is smooth, gets zero.
    
    Attributes:
		name (str):
            Name of the loss. Default: `std_loss`.
        loss_weight (Weights, optional):
			Weight for each loss value. Default: `1.0`. Default: `1.0`.
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
        self.name        = "std_loss"
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
        return self.loss_weight * std_loss(
            input, target,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
        )
