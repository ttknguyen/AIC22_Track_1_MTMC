#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base model for all models defined in `torchkit.models`
"""

from __future__ import annotations

import os
from abc import ABCMeta
from abc import abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Any
from typing import Optional
from typing import Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torchmetrics import Metric

from torchkit.core.data import ClassLabels
from torchkit.core.factory import LOSSES
from torchkit.core.factory import METRICS
from torchkit.core.factory import OPTIMIZERS
from torchkit.core.factory import SCHEDULERS
from torchkit.core.file import create_dirs
from torchkit.core.file import filedir
from torchkit.core.file import is_url_or_file
from torchkit.core.type import Dim3
from torchkit.core.type import EpochOutput
from torchkit.core.type import ForwardOutput
from torchkit.core.type import Indexes
from torchkit.core.type import is_list_of
from torchkit.core.type import Losses_
from torchkit.core.type import Metrics_
from torchkit.core.type import Optimizers_
from torchkit.core.type import Pretrained
from torchkit.core.type import StepOutput
from torchkit.core.type import Tensors
from torchkit.core.utils import console
from torchkit.core.utils import error_console
from torchkit.utils import checkpoints_dir
from torchkit.utils import models_zoo_dir
from torchkit.utils import pretrained_dir
from .debugger import Debugger
from .io import load_pretrained
from .utils import get_next_version

__all__ = [
    "BaseModel",
    "Phase"
]


# MARK: - Phase


class Phase(Enum):
    """3 basic phases of the model: `training`, `testing`, `inference`."""
    
    # Produce predictions, calculate losses and metrics, update weights at the
    # end of each epoch/step.
    TRAINING  = "training"
    # Produce predictions, calculate losses and metrics, DO NOT update weights
    # at the end of each epoch/step.
    TESTING   = "testing"
    # Produce predictions ONLY.
    INFERENCE = "inference"
    
    @staticmethod
    def values() -> list[str]:
        """Return the list of all values."""
        return [e.value for e in Phase]
    
    @staticmethod
    def keys():
        """Return the list of all enum keys."""
        return [e for e in Phase]
    

# MARK: - BaseModel

# noinspection PyAttributeOutsideInit,PyMethodMayBeStatic
class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    """Base model for all models defined in `torchkit.models`. Base model
    only provides access to the attributes. In the model, each head is
    responsible for generating the appropriate output with accommodating loss
    and metric (obviously, we can only calculate specific loss and metric
    with specific output type). So we define the loss functions and metrics
    in the head implementation instead of the model.
    
    Args:
        basename (str, optional):
            Model basename. If `None`, it will be `self.__class__.__name__`.
            Default: `None`.
        name (str, optional):
            Model name. If `None`, it will be `self.__class__.__name__`.
            Default: `None`.
        fullname (str, optional):
            Model fullname in the following format:
            {basename_{name}}_{data_name}_{postfix}. If `None`, it will be
            `self.name`. Default: `None`.
        model_dir (str, optional):
            Model's dir. Default: `None`.
        version (int, str, optional):
            Experiment version. If version is not specified the logger
            inspects the save directory for existing versions, then
            automatically assigns the next available version. If it is a
            string then it is used as the run-specific subdirectory name,
            otherwise `version_${version}` is used. Default: `None`.
        shape (Dim3, optional):
            Image shape as [H, W, C]. Default: `None`.
        num_classes (int, optional):
            Number of classes for classification. Default: `None`.
        class_labels (ClassLabels, optional):
            `ClassLabels` object that contains all labels in the dataset.
             Default: `None`.
        out_indexes (Indexes):
            List of output tensors taken from specific layers' indexes.
            If `>= 0`, return the ith layer's output.
            If `-1`, return the final layer's output.
            Default: `-1`.
        phase (Phase):
            Model's running phase. Default: `Phase.TRAINING`.
        pretrained (Pretrained):
            Initialize weights from pretrained.
            - If `True`, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first
              element in the `model_urls` dictionary.
            - If `str` and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's key
              to get the corresponding local file or url of the weight.
        loss (Losses_, optional):
            Loss function or config(s) to build loss functions. Default: `None`.
        metrics (Metrics_, optional):
            Metric(s) or config(s) to build metric module(s). Default: `None`.
        optimizers (Optimizers_, optional):
            Optimizer(s) or config(s) to build optimizer(s). Default: `None`.
        debugger (dict, optional):
            Debugger's configs. Default: `None`.
            
    Attributes:
        model_zoo (dict):
            A dictionary of all pretrained weights of the model.
        model_dir (str):
            Model's dir. Default: `None`.
        version_dir (str):
            Experiment version dir.
        weights_dir (str):
            Weights dir.
        debug_dir (str):
            Debug output dir.
        epoch_step (int):
            Current step in the epoch. It can be shared between train,
            validation, test, and predict. Mostly used for debugging purpose.
    """
    
    model_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        basename   : Optional[str]             = None,
        name       : Optional[str]             = None,
        fullname   : Optional[str]             = None,
        model_dir  : Optional[str]             = None,
        version    : Optional[Union[int, str]] = None,
        shape      : Optional[Dim3]            = None,
        num_classes: Optional[int] 			   = None,
        class_labels: Optional[ClassLabels]    = None,
        out_indexes: Indexes 				   = -1,
        phase      : Phase                     = Phase.TRAINING,
        pretrained : Pretrained			       = False,
        loss   	   : Optional[Losses_]	   	   = None,
        metrics	   : Optional[Metrics_]	       = None,
        optimizers : Optional[Optimizers_]     = None,
        debugger   : Optional[dict]            = None,
        verbose    : bool                      = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.basename      = basename
        self.name          = name
        self.fullname      = fullname
        self.shape         = shape
        self.num_classes   = num_classes
        self.class_labels  = class_labels
        self.out_indexes   = out_indexes
        self.phase         = phase
        self.pretrained    = pretrained
        self.loss          = loss
        self.train_metrics = None
        self.val_metrics   = None
        self.test_metrics  = None
        self.optims        = optimizers
        self.debugger      = None
        self.verbose       = verbose
        self.epoch_step    = 0
        
        self.init_dirs(model_dir=model_dir, version=version)
        self.init_num_classes()
        self.init_metrics(metrics=metrics)
        self.init_debugger(debugger=debugger)
    
    # MARK: Properties
    
    @property
    def name(self) -> str:
        """Return the model name."""
        return self._name
    
    @name.setter
    def name(self, name: Optional[str] = None):
        """Assign the model name.

        Args:
            name (str, optional):
                Model name. In case `None` is given, it will be
                `self.__class__.__name__`. Default: `None`.
        """
        self._name = (name if (name is not None and name != "")
                      else self.__class__.__name__.lower())
    
    @property
    def fullname(self) -> str:
        """Return the model fullname."""
        return self._fullname
    
    @fullname.setter
    def fullname(self, fullname: Optional[str] = None):
        """Assign the model's fullname in the following format:
        {name}_{data_name}_{postfix}. For instance: `yolov5_coco_1920`.
        In case of `None`, it will be `self.name`.
        """
        self._fullname = (fullname if fullname is not None and fullname != ""
                          else self.name)

    @property
    def basename(self) -> str:
        """Return the model basename. Ex: `scaled_yolov4-p7`, basename is
        `scaled_yolov4`.
        """
        return self._basename

    @basename.setter
    def basename(self, basename: Optional[str] = None):
        """Assign the model's basename. In case of `None`, it will be
        `self.name`.
		"""
        self._basename = (basename if basename is not None and basename != ""
                          else self.name)
    
    @property
    def size(self) -> Optional[Dim3]:
        """Return the input size as [C, H, W]."""
        if self.shape is None:
            return None
        return self.shape[2], self.shape[0], self.shape[1]
    
    @property
    def dim(self) -> Optional[int]:
        """Return the number of dimensions for input."""
        if self.size is None:
            return None
        return len(self.size)
    
    @property
    def ndim(self) -> Optional[int]:
        """Alias of `self.dim()`."""
        return self.dim
    
    @property
    def phase(self) -> Phase:
        """Returns the model's running phase."""
        return self._phase
    
    @phase.setter
    def phase(self, phase: Phase = Phase.TRAINING):
        """Assign the model's running phase."""
        self._phase = phase
        if self._phase is Phase.TRAINING:
            self.unfreeze()
        else:
            self.freeze()
    
    @property
    def pretrained(self) -> Optional[dict]:
        """Returns the model's pretrained metadata."""
        return self._pretrained
    
    @pretrained.setter
    def pretrained(self, pretrained: Pretrained = False):
        """Assign model's pretrained.
        
        Args:
            pretrained (Pretrained):
                Initialize weights from pretrained.
                - If `True`, use the original pretrained described by the
                  author (usually, ImageNet or COCO). By default, it is the
                  first element in the `model_urls` dictionary.
                - If `str` and is a file/path, then load weights from saved
                  file.
                - In each inherited model, `pretrained` can be a dictionary's
                  key to get the corresponding local file or url of the weight.
        """
        if pretrained is True and len(self.model_zoo):
            self._pretrained = list(self.model_zoo.values())[0]
        elif pretrained in self.model_zoo:
            self._pretrained = self.model_zoo[pretrained]
        else:
            self._pretrained = None
        
        # Update num_classes if it is currently `None`.
        if (self._pretrained and self.num_classes is None
            and "num_classes" in self._pretrained):
            self.num_classes = self._pretrained["num_classes"]
            
    @property
    def debug_image_dir(self) -> str:
        """Return the debug image dir path located at: <debug_dir>/<dir>."""
        debug_dir = os.path.join(
            self.debug_dir, f"{self.phase.value}_{(self.current_epoch + 1):03d}"
        )
        create_dirs(paths=[debug_dir])
        return debug_dir
    
    @property
    def debug_image_filepath(self) -> str:
        """Return the debug image filepath located at: <debug_dir>/"""
        save_dir = self.debug_dir
        if self.debugger:
            save_dir = (self.debug_image_dir if self.debugger.save_to_subdir
                        else self.debug_dir)
        
        return os.path.join(
            save_dir,
            f"{self.phase.value}_"
            f"{(self.current_epoch + 1):03d}_"
            f"{(self.epoch_step + 1):06}.jpg"
        )
    
    @property
    def with_loss(self) -> bool:
        """Return whether if the `loss` has been defined."""
        return hasattr(self, "loss") and self.loss is not None
    
    @property
    def loss(self) -> Optional[_Loss]:
        """Return the loss computation module."""
        return self._loss
    
    @loss.setter
    def loss(self, loss: Optional[Losses_]):
        if isinstance(loss, (_Loss, nn.Module)):
            self._loss = loss
        elif isinstance(loss, dict):
            self._loss = LOSSES.build_from_dict(cfg=loss)
        else:
            self._loss = None
        
        # NOTE: Move to device
        """
        if self._loss:
            self._loss.cuda()
        """
    
    @property
    def with_train_metrics(self) -> bool:
        return hasattr(self, "train_metrics") and self.train_metrics is not None
    
    @property
    def train_metrics(self) -> Optional[list[Metric]]:
        return self._train_metrics
    
    @train_metrics.setter
    def train_metrics(self, metrics: Optional[Metrics_]):
        self._train_metrics = self.create_metrics(metrics)
        # This is a simple hack since LightningModule require the
        # metric to be defined with self.<metric>. Here we dynamically
        # add the metric attribute to the class.
        if self._train_metrics:
            for metric in self._train_metrics:
                name = f"train_{metric.name}"
                setattr(self, name, metric)
        
    @property
    def with_val_metrics(self) -> bool:
        return hasattr(self, "val_metrics") and self.val_metrics is not None
    
    @property
    def val_metrics(self) -> Optional[list[Metric]]:
        return self._val_metrics
    
    @val_metrics.setter
    def val_metrics(self, metrics: Optional[Metrics_]):
        self._val_metrics = self.create_metrics(metrics)
        # This is a simple hack since LightningModule require the
        # metric to be defined with self.<metric>. Here we dynamically
        # add the metric attribute to the class
        if self._val_metrics:
            for metric in self._val_metrics:
                name = f"val_{metric.name}"
                setattr(self, name, metric)
    
    @property
    def with_test_metrics(self) -> bool:
        return hasattr(self, "test_metrics") and self.test_metrics is not None
    
    @property
    def test_metrics(self) -> Optional[list[Metric]]:
        return self._test_metrics
    
    @test_metrics.setter
    def test_metrics(self, metrics: Optional[Metrics_]):
        self._test_metrics = self.create_metrics(metrics)
        # This is a simple hack since LightningModule require the
        # metric to be defined with self.<metric>. Here we dynamically
        # add the metric attribute to the class.
        if self._test_metrics:
            for metric in self._test_metrics:
                name = f"test_{metric.name}"
                setattr(self, name, metric)
    
    # MARK: Configure
    
    def init_dirs(
        self, model_dir: Optional[str], version: Optional[Union[int, str]]
    ):
        """Initialize directories.
        
        Args:
            model_dir (str, optional):
                Model's dir. Default: `None`.
            version (int, str, optional):
                Experiment version. If version is not specified the logger
                inspects the save directory for existing versions, then
                automatically assigns the next available version. If it is a
                string then it is used as the run-specific subdirectory name,
                otherwise `version_${version}` is used.
        """
        if model_dir is None:
            self.model_dir = os.path.join(
                checkpoints_dir, self.basename, self.fullname
            )
        else:
            self.model_dir = model_dir
        
        if version is None:
            version = get_next_version(root_dir=self.model_dir)
        if isinstance(version, int):
            version = f"version_{version}"
        self.version = version.lower()
        
        self.version_dir    = os.path.join(self.model_dir,   self.version)
        self.weights_dir    = os.path.join(self.version_dir, "weights")
        self.debug_dir      = os.path.join(self.version_dir, "debugs")
        self.pretrained_dir = os.path.join(pretrained_dir,   self.basename)
        
    def init_num_classes(self):
        """Initialize num_classes."""
        if (
            self.class_labels is not None and
            self.num_classes != self.class_labels.num_classes()
        ):
            self.num_classes = self.class_labels.num_classes()
    
    def init_debugger(self, debugger: Optional[dict]):
        """Initialize debugger."""
        if debugger is None:
            self.debugger = None
        elif isinstance(debugger, dict):
            self.debugger 			= Debugger(**debugger)
            self.debugger.show_func = self.show_results
            """
            self.debug_queue  = Queue(maxsize=self.debug.queue_size)
            self.thread_debug = threading.Thread(
                target=self.show_results_parallel
            )
            """
    
    def init_metrics(self, metrics: Optional[Metrics_]):
        """Initialize all metrics used in the model.
        
        Args:
            One of the 2 options:
            
            - Common metrics for train_/val_/test_metrics:
                "metrics": dict(name="accuracy")
              or,
                "metrics": [dict(name="accuracy"), torchmetrics.Accuracy(),]
            
            - Define train_/val_/test_metrics separately:
                "metrics": {
                    "train": [dict(name="accuracy"), dict(name="f1")]
                    "val":   torchmetrics.Accuracy(),
                    "test":  None,
                }
        """
        if metrics is None:
            self.train_metrics = None
            self.val_metrics   = None
            self.test_metrics  = None
        elif (isinstance(metrics, dict) and
              "train" in metrics or "val" in metrics or "test" in metrics):
            self.train_metrics = metrics.get("train", None)
            self.val_metrics   = metrics.get("val",   None)
            self.test_metrics  = metrics.get("test",  None)
        else:
            self.train_metrics = metrics
            self.val_metrics   = metrics
            self.test_metrics  = metrics
            
    @staticmethod
    def create_metrics(metrics: Optional[Metrics_]) -> Optional[list[Metric]]:
        if isinstance(metrics, Metric):
            return [metrics]
        elif isinstance(metrics, dict):
            return [METRICS.build_from_dict(cfg=metrics)]
        elif isinstance(metrics, list):
            return [METRICS.build_from_dict(cfg=m) if isinstance(m, dict)
                    else m for m in metrics]
        else:
            return None
    
    def load_pretrained(self):
        """Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if self.pretrained:
            filedir.create_dirs([self.pretrained_dir])
            load_pretrained(
                module	  = self,
                model_dir = self.pretrained_dir,
                strict	  = False,
                **self.pretrained
            )
            if self.verbose:
                console.log(f"Load pretrained from: {self.pretrained}!")
        elif is_url_or_file(self.pretrained):
            """load_pretrained(
                self,
                path 	  = self.pretrained,
                model_dir = models_zoo_dir,
                strict	  = False
            )"""
            raise NotImplementedError(
                "This function has not been implemented yet."
            )
        else:
            error_console.log(f"[yellow]Cannot load from pretrained: "
                              f"{self.pretrained}!")
            
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you’d need one. But in the case of GANs or
        similar you might have multiple.

        Returns:
            Any of these 6 options:
                - Single optimizer.
                - List or Tuple of optimizers.
                - Two lists - First list has multiple optimizers, and the
                  second has multiple LR schedulers (or multiple
                  lr_scheduler_config).
                - Dictionary, with an "optimizer" key, and (optionally) a
                  "lr_scheduler" key whose value is a single LR scheduler or
                  lr_scheduler_config.
                - Tuple of dictionaries as described above, with an optional
                  "frequency" key.
                - None - Fit will run without any optimizer.
        """
        optims = self.optims

        if optims is None:
            console.log(f"[yellow]No optimizers have been defined! Consider "
                        f"subclassing this function to manually define the "
                        f"optimizers.")
            return None
        if isinstance(optims, dict):
            optims = [optims]

        # NOTE: List through a list of optimizers
        if not is_list_of(optims, dict):
            raise ValueError
        
        for optim in optims:
            # Define optimizer instance
            optimizer = optim.get("optimizer", None)
            if optimizer is None:
                raise ValueError(f"optimizer must be defined.")
            if isinstance(optimizer, dict):
                optimizer = OPTIMIZERS.build_from_dict(
                    net=self, cfg=optimizer
                )
            optim["optimizer"] = optimizer

            # Define learning rate scheduler
            lr_scheduler = optim.get("lr_scheduler", None)
            if "lr_scheduler" in optim and lr_scheduler is None:
                optim.pop("lr_scheduler")
            elif lr_scheduler is not None:
                scheduler = lr_scheduler.get("scheduler", None)
                if scheduler is None:
                    raise ValueError(f"scheduler must be defined.")
                if isinstance(scheduler, dict):
                    scheduler = SCHEDULERS.build_from_dict(
                        optimizer=optim["optimizer"], cfg=scheduler
                    )
                lr_scheduler["scheduler"] = scheduler
            
            # Define optimizer frequency
            frequency = optim.get("frequency", None)
            if "frequency" in optim and frequency is None:
                optim.pop("frequency")
        
        # NOTE: Re-assign optims
        self.optims = optims
        return self.optims
    
    # MARK: Forward Pass
    
    @abstractmethod
    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor):
                Image of shape [B, C, H, W].
        
        Returns:
            pred (Tensor):
                Predictions.
        """
        pass
    
    @abstractmethod
    def forward_loss(
        self, input: Tensor, target: Tensor, *args, **kwargs
    ) -> ForwardOutput:
        """Forward pass with loss value. Loss function may require more
        arguments beside the ground-truth and prediction values.
        For calculating the metrics, we only need the final predictions and
        ground-truth.

        Args:
            input (Tensor):
                Images image of shape [B, C, H, W].
            target (Tensor):
                Ground-truth image of shape [B, C, H, W].
                
        Returns:
            pred (Tensor):
                Predictions.
            loss (Tensor, optional):
                Loss image.
        """
        pass
    
    def forward_features(
        self, input: Tensor, out_indexes: Optional[Indexes] = None
    ) -> Tensors:
        """Forward pass for features extraction.

        Args:
            input (Tensor):
                Input image.
            out_indexes (Indexes, optional):
                List of layers' indexes to extract features. This is called
                in `forward_features()` and is useful when the model is used as
                a component in another model.
                - If is a `tuple` or `list`, return an array of features.
                - If is a `int`, return only the feature from that layer's
                  index.
                - If is `-1`, return the last layer's output.
                Default: `None`.
        """
        pass
    
    # MARK: Training
    
    def on_fit_start(self):
        """Called at the very beginning of fit."""
        filedir.create_dirs(paths=[
            self.model_dir, self.version_dir, self.weights_dir, self.debug_dir
        ])
        
        if self.debugger:
            self.debugger.run_routine_start()
    
    def on_fit_end(self):
        """Called at the very end of fit."""
        if self.debugger:
            self.debugger.run_routine_end()
            while self.debugger.is_alive():
                continue
    
    def on_train_epoch_start(self):
        """Called in the training loop at the very beginning of the epoch."""
        self.epoch_step = 0
    
    def training_step(
        self, batch: Any, batch_idx: int, *args, **kwargs
    ) -> Optional[StepOutput]:
        """Training step.

        Args:
            batch (Any):
                Batch of inputs. It can be a tuple of
                (`input`, `target`, extra).
            batch_idx (int):
                Batch index.

        Returns:
            outputs (StepOutput, optional):
                - A single loss tensor.
                - A dictionary with the first key must be the `loss`.
                - `None`, training will skip to the next batch.
        """
        # NOTE: Forward pass
        input, target, extra = batch[0], batch[1], batch[2:]
        pred, loss = self.forward_loss(
            input=input, target=target, *args, **kwargs
        )
        return {"loss": loss, "input": input, "target": target, "pred": pred}
    
    def training_step_end(
        self, outputs: Optional[StepOutput], *args, **kwargs
    ) -> Optional[StepOutput]:
        """Use this when training with dp or ddp2 because training_step() will
        operate on only part of the batch. However, this is still optional and
        only needed for things like softmax or NCE loss.
        
        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        """
        if not isinstance(outputs, dict):
            return None
        
        # NOTE: Gather results
        # For DDP strategy
        if self.trainer.num_processes > 1:
            outputs = self.all_gather(outputs)
        
        losses = outputs["loss"]    # losses from each GPU
        input  = outputs["input"]   # images from each GPU
        target = outputs["target"]  # ground-truths from each GPU
        pred   = outputs["pred"]    # predictions from each GPU
        
        # NOTE: Tensors
        if self.trainer.num_processes > 1:
            input  = input.flatten(start_dim=0, end_dim=1)
            target = target.flatten(start_dim=0, end_dim=1)
            pred   = pred.flatten(start_dim=0, end_dim=1)
        
        # NOTE: Loss
        loss = losses.mean() if losses is not None else None
        self.ckpt_log_scalar(f"checkpoint/loss/train_step", loss)
        # self.tb_log(f"{loss_tag}", loss, "step")
       
        # NOTE: Metrics
        if self.with_train_metrics:
            for i, metric in enumerate(self.train_metrics):
                value = metric(pred, target)
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/train_step", value, True)
                # self.tb_log(f"{metric.name}/train_step", value, "step")
        
        self.epoch_step += 1
        return {"loss": loss}
    
    def training_epoch_end(self, outputs: EpochOutput):
        # NOTE: Loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.ckpt_log_scalar(f"checkpoint/loss/train_epoch", loss)
        self.tb_log_scalar(f"loss/train_epoch", loss, "epoch")
        
        # NOTE: Metrics
        if self.with_train_metrics:
            for i, metric in enumerate(self.train_metrics):
                value = metric.compute()
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/train_epoch", value)
                self.tb_log_scalar(f"{metric.name}/train_epoch", value, "epoch")
                metric.reset()
    
    def on_validation_epoch_start(self):
        """Called in the validation loop at the very beginning of the epoch."""
        self.epoch_step = 0
        
    def validation_step(
        self, batch: Any, batch_idx: int, *args, **kwargs
    ) -> Optional[StepOutput]:
        """Validation step.

        Args:
            batch (Any):
                Batch of inputs. It can be a tuple of
                (`input`, `target`, extra).
            batch_idx (int):
                Batch index.

        Returns:
            outputs (StepOutput, optional):
                - A single loss image.
                - A dictionary with the first key must be the `loss`.
                - `None`, training will skip to the next batch.
        """
        input, target, extra = batch[0], batch[1], batch[2:]
        pred, loss = self.forward_loss(
            input=input, target=target, *args, **kwargs
        )
        return {"loss": loss, "input": input, "target": target, "pred": pred}
        
    def validation_step_end(
        self, outputs: Optional[StepOutput], *args, **kwargs
    ) -> Optional[StepOutput]:
        """Use this when validating with dp or ddp2 because `validation_step`
        will operate on only part of the batch. However, this is still optional
        and only needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        """
        if not isinstance(outputs, dict):
            return None
        
        # NOTE: Gather results
        # For DDP strategy
        if self.trainer.num_processes > 1:
            outputs = self.all_gather(outputs)
        
        losses = outputs["loss"]    # losses from each GPU
        input  = outputs["input"]   # images from each GPU
        target = outputs["target"]  # ground-truths from each GPU
        pred   = outputs["pred"]    # predictions from each GPU
        
        # NOTE: Tensors
        if self.trainer.num_processes > 1:
            input  = input.flatten(start_dim=0, end_dim=1)
            target = target.flatten(start_dim=0, end_dim=1)
            pred   = pred.flatten(start_dim=0, end_dim=1)
            
        # NOTE: Debugging
        epoch = self.current_epoch + 1
        if (self.debugger and epoch % self.debugger.every_n_epochs == 0 and
            self.epoch_step < self.debugger.save_max_n):
            if self.trainer.is_global_zero:
                self.debugger.run(
                    deepcopy(input), deepcopy(target), deepcopy(pred),
                    self.debug_image_filepath
                )

        # NOTE: Loss
        loss = losses.mean() if losses is not None else None
        self.ckpt_log_scalar(f"checkpoint/loss/val_step", loss)
        # self.tb_log(f"{loss_tag}", loss, "step")
        
        # NOTE: Metrics
        if self.with_val_metrics:
            for i, metric in enumerate(self.val_metrics):
                value = metric(pred, target)
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/val_step", value)
                # self.tb_log(f"{metric.name}/val_step", value, "step")
            
        self.epoch_step += 1
        return {"loss": loss}

    def validation_epoch_end(self, outputs: EpochOutput):
        # NOTE: Loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.ckpt_log_scalar(f"checkpoint/loss/val_epoch", loss)
        self.tb_log_scalar(f"loss/val_epoch", loss, "epoch")
        
        # NOTE: Metrics
        if self.with_val_metrics:
            for i, metric in enumerate(self.val_metrics):
                value = metric.compute()
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/val_epoch", value)
                self.tb_log_scalar(f"{metric.name}/val_epoch", value, "epoch")
                metric.reset()
                
    def on_test_start(self) -> None:
        """Called at the very beginning of testing."""
        filedir.create_dirs(paths=[
            self.model_dir, self.version_dir, self.weights_dir, self.debug_dir
        ])
        
        if self.debugger:
            self.debugger.run_routine_start()
        
    def on_test_epoch_start(self) -> None:
        """Called in the test loop at the very beginning of the epoch."""
        self.epoch_step = 0
    
    def test_step(
        self, batch: Any, batch_idx: int, *args, **kwargs
    ) -> Optional[StepOutput]:
        """Test step.

        Args:
            batch (Any):
                Batch of inputs. It can be a tuple of (`input`, `target`, extra).
            batch_idx (int):
                Batch index.

        Returns:
            outputs (StepOutput, optional):
                - A single loss image.
                - A dictionary with the first key must be the `loss`.
                - `None`, training will skip to the next batch.
        """
        input, target, extra = batch[0], batch[1], batch[2:]
        pred, loss = self.forward_loss(
            input=input, target=target, *args, **kwargs
        )
        return {"loss": loss, "input": input, "target": target, "pred": pred}
    
    def test_step_end(
        self, outputs: Optional[StepOutput], *args, **kwargs
    ) -> Optional[StepOutput]:
        """Use this when testing with dp or ddp2 because `test_step` will
        operate on only part of the batch. However, this is still optional and
        only needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        """
        if not isinstance(outputs, dict):
            return None
        
        # NOTE: Gather results
        # For DDP strategy
        if self.trainer.num_processes > 1:
            outputs = self.all_gather(outputs)
        
        losses = outputs["loss"]    # losses from each GPU
        input  = outputs["input"]   # images from each GPU
        target = outputs["target"]  # ground-truths from each GPU
        pred   = outputs["pred"]    # predictions from each GPU
        
        # NOTE: Tensors
        if self.trainer.num_processes > 1:
            input  = input.flatten(start_dim=0, end_dim=1)
            target = target.flatten(start_dim=0, end_dim=1)
            pred   = pred.flatten(start_dim=0, end_dim=1)
        
        # NOTE: Loss
        loss = losses.mean() if losses is not None else None
        self.ckpt_log_scalar(f"checkpoint/loss/test_step", loss)
        # self.tb_log(f"loss/test_step", loss, "step")
        
        # NOTE: Metrics
        if self.with_test_metrics:
            for i, metric in enumerate(self.test_metrics):
                value = metric(pred, target)
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/test_step", value)
                # self.tb_log(f"{metric.name}/test_step", value, "step")
        
        self.epoch_step += 1
        return {"loss": loss}

    def test_epoch_end(self, outputs: EpochOutput):
        # NOTE: Loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.ckpt_log_scalar(f"checkpoint/loss/test_epoch", loss)
        self.tb_log_scalar(f"loss/test_epoch", loss, "epoch")

        # NOTE: Metrics
        if self.with_test_metrics:
            for i, metric in enumerate(self.test_metrics):
                value = metric.compute()
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/test_epoch", value)
                self.tb_log_scalar(f"{metric.name}/test_epoch", value, "epoch")
                metric.reset()
    
    # MARK: Export
    
    def export_to_onnx(
        self,
        input_dims   : Optional[Dim3] = None,
        filepath     : Optional[str]  = None,
        export_params: bool           = True
    ):
        """Export the model to `onnx` format.

        Args:
            input_dims (Dim3, optional):
                Input dimensions. Default: `None`.
            filepath (str, optional):
                Path to save the model. If `None` or empty, then save to
                `zoo_dir`. Default: `None`.
            export_params (bool):
                Should export parameters also? Default: `True`.
        """
        # NOTE: Check filepath
        if filepath in [None, ""]:
            filepath = os.path.join(self.version_dir, f"{self.fullname}.onnx")
        if ".onnx" not in filepath:
            filepath += ".onnx"
        
        if input_dims is not None:
            input_sample = torch.randn(input_dims)
        elif self.dims is not None:
            input_sample = torch.randn(self.dims)
        else:
            raise ValueError(f"No input dims are defined.")
        
        self.to_onnx(filepath=filepath, input_sample=input_sample,
                     export_params=export_params)
    
    def export_to_torchscript(
        self,
        input_dims: Optional[Dim3] = None,
        filepath  : Optional[str]  = None,
        method    : str            = "script"
    ):
        """Export the model to `TorchScript` format.

        Args:
            input_dims (Dim3, optional):
                Input dimensions.
            filepath (str, optional):
                Path to save the model. If `None` or empty, then save
                to `zoo_dir`. Default: `None`.
            method (str):
                Whether to use TorchScript's `script` or `trace` method.
                Default: `script`
        """
        # NOTE: Check filepath
        if filepath in [None, ""]:
            filepath = os.path.join(self.version_dir, f"{self.fullname}.pt")
        if ".pt" not in filepath:
            filepath += ".pt"
        
        if input_dims is not None:
            input_sample = torch.randn(input_dims)
        elif self.dims is not None:
            input_sample = torch.randn(self.dims)
        else:
            raise ValueError(f"No input dims are defined.")
        
        script = self.to_torchscript(method=method, example_inputs=input_sample)
        torch.jit.save(script, filepath)

    # MARK: Visualize
    
    @abstractmethod
    def show_results(
        self,
        input    	 : Optional[Tensor] = None,
        target	     : Optional[Tensor] = None,
        pred		 : Optional[Tensor] = None,
        filepath     : Optional[str]    = None,
        image_quality: int              = 95,
        verbose      : bool             = False,
        show_max_n   : int              = 8,
        wait_time    : float            = 0.01,
        *args, **kwargs
    ):
        """Show results.

        Args:
            input (Tensor, optional):
                Input images.
            target (Tensor, optional):
                Ground-truth.
            pred (Tensor, optional):
                Predictions.
            filepath (str, optional):
                File path to save the debug result.
            image_quality (int):
                Image quality to be saved. Default: `95`.
            verbose (bool):
                If `True` shows the results on the screen. Default: `False`.
            show_max_n (int):
                Maximum debugging items to be shown. Default: `8`.
            wait_time (float):
                Pause some times before showing the next image.
        """
        pass

    # MARK: Log
    
    def tb_log_scalar(
        self, tag: str, data: Optional[Any], step: Union[str, int] = "step"
    ):
        """Log scalar values using tensorboard."""
        if data is None:
            return
        if isinstance(step, str):
            step = self.current_epoch if step == "epoch" else self.global_step
        if self.trainer.is_global_zero:
            self.logger.experiment.add_scalar(tag, data, step)
    
    def tb_log_class_metrics(
        self, tag: str, data: Optional[Any], step: Union[str, int] = "step"
    ):
        """Log class metrics using tensorboard."""
        if data is None:
            return
        if self.class_labels is None:
            return
        if isinstance(step, str):
            step = self.current_epoch if step == "epoch" else self.global_step
        if self.trainer.is_global_zero:
            for n, a in zip(self.class_labels.names(), data):
                n = f"{tag}/{n}"
                self.logger.experiment.add_scalar(n, a, step)
        
    def ckpt_log_scalar(
        self, tag: str, data: Optional[Any], prog_bar: bool = False
    ):
        """Log for model checkpointing."""
        if data is None:
            return
        if self.trainer.is_global_zero:
            self.log(tag, data, prog_bar=prog_bar, sync_dist=True,
                     rank_zero_only=True)
