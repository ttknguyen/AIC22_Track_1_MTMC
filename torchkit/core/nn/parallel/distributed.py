#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
import platform
from typing import Union

import torch
from pytorch_lightning.plugins import DDP2Plugin
from pytorch_lightning.plugins import DDPPlugin

from torchkit.core.type import Callable
from torchkit.core.utils import console

__all__ = [
	"set_distributed_backend",
]


def set_distributed_backend(strategy: Union[str, Callable], cudnn: bool = True):
	# NOTE: cuDNN
	if torch.backends.cudnn.is_available():
		torch.backends.cudnn.enabled = cudnn
		console.log(
			f"cuDNN available: [bright_green]True[/bright_green], "
			f"used:" + "[bright_green]True" if cudnn else "[red]False"
		)
	else:
		console.log(f"cuDNN available: [red]False")
	
	# NOTE: Torch Distributed Backend
	if strategy in ["ddp", "ddp2"] or isinstance(strategy, (DDPPlugin, DDP2Plugin)):
		if platform.system() == "Windows":
			os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
			console.log(
				"Running on a Windows machine, set torch distributed backend "
				"to gloo."
			)
		elif platform.system() == "Linux":
			os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
			console.log(
				"Running on a Unix machine, set torch distributed backend "
				"to nccl."
			)
