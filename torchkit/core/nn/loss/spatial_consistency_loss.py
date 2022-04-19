#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Spatial Consistency Loss.

The spatial consistency loss encourages spatial coherence of the enhanced
image through preserving the difference of neighboring regions between the
input image and its enhanced version.

References:
    https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from torchkit.core.factory import LOSSES
from torchkit.core.type import Tensors
from torchkit.core.type import Weights
from .utils import weighted_loss
from .utils import weighted_sum

__all__ = [
    "elementwise_spatial_consistency_loss",
    "elementwise_spa_loss",
    "spatial_consistency_loss",
    "spa_loss",
    "SpatialConsistencyLoss",
    "SPALoss",
]

kernel_left  = torch.FloatTensor([[0,  0, 0], [-1, 1,  0], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
kernel_right = torch.FloatTensor([[0,  0, 0], [ 0, 1, -1], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
kernel_up    = torch.FloatTensor([[0, -1, 0], [ 0, 1,  0], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
kernel_down  = torch.FloatTensor([[0,  0, 0], [ 0, 1,  0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

weight_left  = nn.Parameter(data=kernel_left,  requires_grad=False)
weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
weight_up    = nn.Parameter(data=kernel_up,    requires_grad=False)
weight_down  = nn.Parameter(data=kernel_down,  requires_grad=False)
pool         = nn.AvgPool2d(4)


# MARK: - SpatialConsistencyLoss

@weighted_loss
def elementwise_spatial_consistency_loss(
    input: Tensor, target: Optional[Tensor], pred: Tensor
) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    global weight_left, weight_right, weight_up, weight_down, pool
    
    if weight_left.device != input.device:
        weight_left = weight_left.to(input.device)
    if weight_right.device != input.device:
        weight_right = weight_right.to(input.device)
    if weight_up.device != input.device:
        weight_up = weight_up.to(input.device)
    if weight_down.device != input.device:
        weight_down = weight_down.to(input.device)
    
    input_mean      = torch.mean(input, 1, keepdim=True)
    pred_mean       = torch.mean(pred,  1, keepdim=True)

    input_pool      = pool(input_mean)
    pred_pool       = pool(pred_mean)

    d_org_left      = F.conv2d(input_pool, weight_left,  padding=1)
    d_org_right     = F.conv2d(input_pool, weight_right, padding=1)
    d_org_up        = F.conv2d(input_pool, weight_up,    padding=1)
    d_org_down      = F.conv2d(input_pool, weight_down,  padding=1)

    d_enhance_left  = F.conv2d(pred_pool, weight_left,  padding=1)
    d_enhance_right = F.conv2d(pred_pool, weight_right, padding=1)
    d_enhance_up    = F.conv2d(pred_pool, weight_up,    padding=1)
    d_enhance_down  = F.conv2d(pred_pool, weight_down,  padding=1)

    d_left          = torch.pow(d_org_left  - d_enhance_left,  2)
    d_right         = torch.pow(d_org_right - d_enhance_right, 2)
    d_up            = torch.pow(d_org_up    - d_enhance_up,    2)
    d_down          = torch.pow(d_org_down  - d_enhance_down,  2)
    loss            = d_left + d_right + d_up + d_down
    
    return loss
    

def spatial_consistency_loss(
    input             : Tensors,
    pred              : Tensors,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Weights] = None,
    reduction         : str               = "mean",
) -> Tensor:
    """Function that takes the mean element-wise absolute value difference.
    
    Args:
        input (Tensors):
            Prediction image of shape [B, C, H, W].
        pred (Tensors):
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
    
    if isinstance(pred, Tensor):  # Single output
        pred = [pred]
    elif isinstance(pred, dict):
        pred = list(pred.values())
    if not isinstance(pred, (list, tuple)):
        raise ValueError(f"Do not support pred of type: {type(pred)}.")
    
    if len(input) != len(pred):
        raise ValueError(f"Length of input and pred must be equal."
                         f"But got: {len(input)} != {len(pred)}")
    
    losses = [
        elementwise_spatial_consistency_loss(
            input=i, pred=p, weight=elementwise_weight, reduction=reduction
        ) for i, p in zip(input, pred)
    ]
    return weighted_sum(losses, input_weight)
   
   
@LOSSES.register(name="spatial_consistency_loss")
class SpatialConsistencyLoss(_Loss):
    """Spatial Consistency Loss (SPA) Loss.

    Attributes:
        name (str):
            Name of the loss. Default: `spatial_consistency_loss`.
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
        self.name        = "spatial_consistency_loss"
        self.loss_weight = loss_weight
        
        if self.reduction not in self.reductions:
            raise ValueError(f"Supported reduction are: {self.reductions}. "
                             f"But got: {self.reduction}.")
 
    # MARK: Forward Pass
    
    def forward(
        self,
        input             : Tensors,
        pred              : Tensors,
        input_weight      : Optional[Weights] = None,
        elementwise_weight: Optional[Weights] = None,
        **_
    ) -> Tensor:
        return self.loss_weight * spatial_consistency_loss(
            input              = input,
            pred               = pred,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
        )


# MARK: - SPALoss

elementwise_spa_loss = elementwise_spatial_consistency_loss
spa_loss             = spatial_consistency_loss
SPALoss              = SpatialConsistencyLoss

LOSSES.register(name="spa_loss", module=SPALoss)
