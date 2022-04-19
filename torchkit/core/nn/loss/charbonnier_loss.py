#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Charbonnier Loss.
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
    "elementwise_charbonnier_loss",
    "charbonnier_loss",
    "CharbonnierLoss",
]


# MARK: - CharbonnierLoss

@weighted_loss
def elementwise_charbonnier_loss(
    input: Tensor, target: Tensor, eps: float = 1e-3
) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    return torch.sqrt((input - target) ** 2 + (eps * eps))


def charbonnier_loss(
    input             : Tensors,
    target            : Tensor,
    eps               : float             = 1e-3,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Weights] = None,
    reduction         : str               = "mean"
) -> Tensor:
    """

    Args:
        input (Tensors):
            Prediction image of shape [B, C, H, W].
        target (Tensor):
            Ground-truth image of shape [B, C, H, W].
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
    """
    if isinstance(input, Tensor):  # Single output
        input = [input]
    elif isinstance(input, dict):
        input = list(input.values())
    if not isinstance(input, (list, tuple)):
        raise ValueError(f"Do not support input of type: {type(input)}.")
    
    losses = [
        elementwise_charbonnier_loss(
            input=i, target=target, eps=eps, weight=elementwise_weight,
            reduction=reduction
        ) for i in input
    ]
    return weighted_sum(losses, input_weight)


@LOSSES.register(name="charbonnier_loss")
class CharbonnierLoss(_Loss):
    """Charbonnier Loss. Supports both single output and multi-outputs input.
    
    Attributes:
        name (str):
            Name of the loss. Default: `charbonnier_loss`.
        eps (float):
            Small value for numerically stability when dividing.
            Default: `1e-3`.
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
        eps        : float             = 1e-3,
        loss_weight: Optional[Weights] = 1.0,
        reduction  : str               = "mean"
    ):
        super().__init__(reduction=reduction)
        self.name        = "charbonnier_loss"
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
        # diff = input - target
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        # loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return self.loss_weight * charbonnier_loss(
			input, target, self.eps,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
		)
