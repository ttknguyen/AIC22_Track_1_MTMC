#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perceptual loss.
"""

from __future__ import annotations

from typing import Optional

from torch import nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from torchkit.core import LOSSES
from torchkit.core.type import Tensors
from torchkit.core.type import Weights
from .mae_loss import L1Loss
from .perceptual_loss import PerceptualLoss

__all__ = [
    "PerceptualL1Loss",
]

   
# MARK: - PerceptualL1Loss

@LOSSES.register(name="perceptual_l1_Loss")
class PerceptualL1Loss(_Loss):
    """Loss = weights[0] * Perceptual Loss + weights[1] * L1 Loss
    """

    reductions = ["none", "mean", "sum"]
    
    def __init__(
        self,
        vgg        : nn.Module,
        loss_weight: Optional[Weights] = Tensor([1.0, 1.0]),
        reduction  : str               = "mean",
    ):
        super().__init__(reduction=reduction)
        self.name        = "perceptual_l1_Loss"
        self.loss_weight = loss_weight
        self.per_loss    = PerceptualLoss(vgg=vgg, reduction=reduction)
        self.l1_loss     = L1Loss(reduction=reduction)
        self.layer_name_mapping = {
            "3" : "relu1_2",
            "8" : "relu2_2",
            "15": "relu3_3"
        }
        
        if self.loss_weight is None:
            self.loss_weight = Tensor([1.0, 1.0])
        elif len(self.loss_weight) != 2:
            raise ValueError(f"loss_weight must has length == 2.")
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
        l0 = self.loss_weight[0] * self.per_loss(input, target, input_weight, elementwise_weight)
        l1 = self.loss_weight[1] * self.l1_loss(input, target, input_weight, elementwise_weight)
        return l0 + l1
