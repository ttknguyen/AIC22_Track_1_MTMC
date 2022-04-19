#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perceptual loss.
"""

from __future__ import annotations

from typing import Optional

from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from torchkit.core import LOSSES
from torchkit.core.nn.loss.utils import weighted_sum
from torchkit.core.type import Tensors
from torchkit.core.type import Weights

__all__ = [
    "PerceptualLoss",
]


# MARK: - PerceptualLoss

@LOSSES.register(name="perceptual_Loss")
class PerceptualLoss(_Loss):
    
    reductions = ["none", "mean", "sum"]
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        vgg        : nn.Module,
        loss_weight: Optional[Weights] = 1.0,
        reduction  : str               = "mean"
    ):
        super().__init__(reduction=reduction)
        self.name        = "perceptual_Loss"
        self.loss_weight = loss_weight
        self.vgg         = vgg
        self.vgg.freeze()
        
        if self.reduction not in self.reductions:
            raise ValueError(f"Supported reduction are: {self.reductions}. "
                             f"But got: {self.reduction}.")

    # MARK: Forward Pass

    def forward(
        self,
        input       : Tensors,
        target      : Tensor,
        input_weight: Optional[Weights] = None,
        **_
    ) -> Tensor:
        if isinstance(input, Tensor):  # Single output
            input = [input]
        elif isinstance(input, dict):
            input = list(input.values())
        if not isinstance(input, (list, tuple)):
            raise ValueError(f"Do not support input of type: {type(input)}.")

        if self.vgg.device != input[0].device:
            self.vgg = self.vgg.to(input[0].device)

        losses = []
        for i in input:
            input_features  = self.vgg.forward_features(i)
            target_features = self.vgg.forward_features(target)
            loss = [
                F.mse_loss(i_feature, t_feature)
                for i_feature, t_feature in zip(input_features, target_features)
            ]
            losses.append(sum(loss) / len(loss))
            
        return self.loss_weight * weighted_sum(losses, input_weight)
