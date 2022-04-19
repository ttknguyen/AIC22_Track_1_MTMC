#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Template for (Supervised) 2D Object Detection datasets.
"""

from __future__ import annotations

import os
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Optional

import numpy as np
import torch
from joblib import delayed
from joblib import Parallel
from sortedcontainers import SortedDict
from torch import Tensor
from torchvision.datasets import VisionDataset

from torchkit.core.data.augment import BaseAugment
from torchkit.core.data.data_class import ClassLabels
from torchkit.core.data.data_class import ImageInfo
from torchkit.core.data.data_class import VisionData
from torchkit.core.data.handler import VisionDataHandler
from torchkit.core.factory import AUGMENTS
from torchkit.core.file import create_dirs
from torchkit.core.file import get_hash
from torchkit.core.type import Augment_
from torchkit.core.type import Dim3
from torchkit.core.utils import console
from torchkit.core.utils import download_bar
from torchkit.core.utils import progress_bar
from torchkit.core.vision import bbox_cxcywh_norm_to_xyxy
from torchkit.core.vision import get_image_size
from torchkit.core.vision import letterbox_resize
from torchkit.core.vision import read_image
from torchkit.core.vision import resize
from torchkit.core.vision import shift_bbox
from torchkit.core.vision import to_tensor
from torchkit.core.vision import VISION_BACKEND
from torchkit.core.vision import VisionBackend

__all__ = [
    "ObjectDetectionDataset"
]


# MARK: - ObjectDetectionDataset

class ObjectDetectionDataset(VisionDataset, metaclass=ABCMeta):
    """A base class for all (supervised) 2D Object Detection datasets.
    
    Attributes:
        root (str):
            Dataset root directory that contains: train/val/test/...
            subdirectories.
        split (str):
            Split to use. One of: ["train", "val", "test"].
        image_paths (list):
            List of all image files.
        label_paths (list):
            List of all label files.
         custom_label_paths (list):
            List of all custom label files.
        data (list[VisionData]):
            List of all `VisionData` objects.
        class_labels (ClassLabels, optional):
            `ClassLabels` object contains all class-labels defined in the
            dataset.
        shape (Dim3):
            Image shape as [H, W, C]
        batch_size (int):
            Number of training samples in one forward & backward pass.
        batch_shapes (np.ndarray, optional):
            Array of batch shapes. It is available only for `rect_training`
            augmentation.
        batch_indexes (np.ndarray, optional):
            Array of batch indexes. It is available only for `rect_training`
            augmentation.
        caching_labels (bool):
            Should overwrite the existing cached labels?
        caching_images (bool):
            Cache images into memory for faster training.
        write_labels (bool):
            After loading images and labels for the first time, we will convert
            it to our custom data format and write to files. If `True`, we will
            overwrite these files.
        fast_dev_run (bool):
            Take a small subset of the data for fast debug (i.e, like unit
            testing).
        load_augment (dict):
            Augmented loading policy.
        augment (Augment_):
            Augmentation policy.
        transforms (callable, optional):
            Function/transform that takes input sample and its target as
            entry and returns a transformed version.
        transform (callable, optional):
            Function/transform that takes input sample as entry and returns
            a transformed version.
        target_transform (callable, optional):
            Function/transform that takes in the target and transforms it.
            
    Work Flow:
        __init__()
            |__ list_files()
            |__ load_data()
            |		|__ cache_labels()
            |		|		|__ load_label()
            |		|		|__ load_labels()
            |		|__ cache_images()
            |				|__ load_images()
            |__ post_load_data()
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        split           : str,
        class_labels    : Optional[ClassLabels]   = None,
        shape           : Dim3                    = (640, 640, 3),
        batch_size      : int                     = 1,
        caching_labels  : bool                    = False,
        caching_images  : bool                    = False,
        write_labels    : bool                    = False,
        fast_dev_run    : bool                    = False,
        load_augment    : Optional[dict]          = None,
        augment         : Optional[Augment_]      = None,
        vision_backend  : Optional[VisionBackend] = None,
        transforms      : Optional[Callable]      = None,
        transform       : Optional[Callable]      = None,
        target_transform: Optional[Callable]      = None,
        *args, **kwargs
    ):
        super().__init__(
            root			 = root,
            transforms       = transforms,
            transform        = transform,
            target_transform = target_transform
        )
        self.split              = split
        self.class_labels       = class_labels
        self.shape              = shape
        self.batch_size         = batch_size
        self.batch_shapes       = None
        self.batch_indexes      = None
        self.caching_labels     = caching_labels
        self.caching_images     = caching_images
        self.write_labels       = write_labels
        self.fast_dev_run       = fast_dev_run
        self.load_augment       = load_augment
        self.augment 		    = augment
        
        if vision_backend in VisionBackend:
            self.vision_backend = vision_backend
        else:
            self.vision_backend = VISION_BACKEND
        
        self.image_paths        = []
        self.label_paths        = []
        self.custom_label_paths = []
        self.data               = []

        # NOTE: List files
        self.list_files()
        # NOTE: Load class_labels
        self.load_class_labels()
        # NOTE: Load (and cache) data
        self.load_data()
        # NOTE: Post-load data
        self.post_load_data()
        
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Any:
        """Return a tuple of data item from the dataset."""
        items  = self.get_item(index=index)
        input  = items[0]
        target = items[1]
        rest   = items[2:]
        
        if self.transform:
            input  = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        if self.transforms:
            input  = self.transforms(input)
            target = self.transforms(target)
        return input, target, rest
    
    # MARK: Properties
    
    @property
    def augment(self) -> Optional[BaseAugment]:
        return self._augment
    
    @augment.setter
    def augment(self, augment: Optional[Augment_]):
        """Assign augment configs."""
        if isinstance(augment, BaseAugment):
            self._augment = augment
        elif isinstance(augment, dict):
            self._augment = AUGMENTS.build_from_dict(cfg=augment)
        else:
            self._augment = None
    
    @property
    def image_size(self) -> int:
        """Return image size."""
        return max(self.shape)
    
    @property
    def has_custom_labels(self) -> bool:
        """Check if we have custom label files. If `True`, then those files
        will be loaded. Else, load the raw data	from the dataset, convert
        them to our custom data format, and write to files.
        """
        """
        n = len(self.custom_label_paths)
        return (
            n > 0 and
            n == len(self.image_paths) and
            all(os.path.isfile(p) for p in self.custom_label_paths)
        )
        """
        return self._has_custom_labels
    
    @property
    def custom_label_paths(self) -> list:
        return self._custom_label_paths
    
    @custom_label_paths.setter
    def custom_label_paths(self, paths: Optional[list[int]]):
        self._custom_label_paths = paths
        
        n = len(self.custom_label_paths)
        self._has_custom_labels = (
            n > 0 and
            n == len(self.image_paths) and
            all(os.path.isfile(p) for p in self.custom_label_paths)
        )
    
    # MARK: List Files
    
    @abstractmethod
    def list_files(self):
        """List image and label files.
        
        Todos:
            - Look for image and label files in `split` directory.
            - We should look for our custom label files first.
            - If none is found, proceed to listing the images and raw labels'
              files.
            - After this method, these following attributes MUST be defined:
              `image_paths`, `label_paths`, `custom_label_paths`.
        """
        pass
    
    # MARK: Load ClassLabels

    @abstractmethod
    def load_class_labels(self):
        """Load ClassLabels."""
        pass

    # MARK: Load Data
    
    def load_data(self):
        """Load labels, cache labels and images."""
        # NOTE: Cache labels
        """
        file = (
            self.label_paths if isinstance(self.label_paths, str)
            else self.label_paths[0]
        )
        split_prefix      = file[: file.find(self.split)]
        cached_label_path = f"{split_prefix}{self.split}.cache"
        """
        cached_label_path = os.path.join(self.root, f"{self.split}.cache")
        
        if os.path.isfile(cached_label_path):
            cache = torch.load(cached_label_path)  # Load
            hash  = (get_hash([self.label_paths] + self.custom_label_paths + self.image_paths)
                     if isinstance(self.label_paths, str)
                     else get_hash(self.label_paths + self.custom_label_paths + self.image_paths))
            if self.caching_labels or cache["hash"] != hash:
                # Re-cache
                cache = self.cache_labels_multithreaded(path=cached_label_path)
        else:
            # Cache
            cache = self.cache_labels_multithreaded(path=cached_label_path)
        
        # NOTE: Get labels
        self.data = [cache[x] for x in self.image_paths]

        # NOTE: Cache images
        if self.caching_images:
            self.cache_images()
        
    def cache_labels(self, path: str) -> dict:
        """Cache labels, check images and read shapes.
        
        Args:
            path (str):
                Path to save the cached labels.
        
        Returns:
            cache_labels (dict):
                Dictionary contains the labels (numpy array) and the
                original image shapes that were cached.
        """
        # NOTE: Load all labels from a `.json` file
        if isinstance(self.label_paths, str) and os.path.isfile(self.label_paths):
            cache_labels = self.load_labels(self.image_paths, self.label_paths)
        # NOTE: Load each pair of image file and label file together
        else:
            cache_labels = {}

            with progress_bar() as pbar:
                for i in pbar.track(
                    range(len(self.image_paths)),
                    description=f"Caching {self.split} labels"
                ):
                    label_path = (
                        self.label_paths[i] if os.path.isfile(self.label_paths[i])
                        else None
                    )
                    custom_label_path = (
                        self.custom_label_paths[i]
                        if (self.has_custom_labels
                            and os.path.isfile(self.custom_label_paths[i]))
                        else None
                    )
                    label = self.load_label(
                        image_path 	 	  = self.image_paths[i],
                        label_path	 	  = label_path,
                        custom_label_path = custom_label_path
                    )
                    cache_labels[self.image_paths[i]] = label
            
        # NOTE: Check for any changes btw the cached labels
        for (k, v) in cache_labels.items():
            if v.image_info.path != k:
                self.caching_labels = True
                break
        
        # NOTE: Write cache
        console.log(f"Labels has been cached to: {path}.")
        if isinstance(self.label_paths, str):
            cache_labels["hash"] = get_hash(
                [self.label_paths] + self.custom_label_paths + self.image_paths
            )
        else:
            cache_labels["hash"] = get_hash(
                self.label_paths + self.custom_label_paths + self.image_paths
            )
        torch.save(cache_labels, path)  # Save for next time
        return cache_labels
    
    def cache_labels_multithreaded(self, path: str) -> dict:
        """Cache labels, check images and read shapes with multi-threading.
        
        Args:
            path (str):
                Path to save the cached labels.
        
        Returns:
            cache_labels (dict):
                Dictionary contains the labels (numpy array) and the
                original image shapes that were cached.
        """
        # NOTE: Load all labels from a `.json` file
        if isinstance(self.label_paths, str) and os.path.isfile(self.label_paths):
            cache_labels = self.load_labels(self.image_paths, self.label_paths)
        # NOTE: Load each pair of image file and label file together
        else:
            cache_labels = {}
            total        = len(self.image_paths)
        
            with progress_bar() as pbar:
                task = pbar.add_task(
                    f"[bright_yellow]Caching {self.split} labels", total=total
                )
                
                def cache_label(i):
                    label_path = (
                        self.label_paths[i] if os.path.isfile(self.label_paths[i])
                        else None
                    )
                    custom_label_path = (
                        self.custom_label_paths[i]
                        if (self.has_custom_labels
                            and os.path.isfile(self.custom_label_paths[i]))
                        else None
                    )
                    label = self.load_label(
                        image_path 	 	  = self.image_paths[i],
                        label_path	 	  = label_path,
                        custom_label_path = custom_label_path
                    )
                    cache_labels[self.image_paths[i]] = label
                    pbar.update(task, advance=1)
                
                Parallel(n_jobs=os.cpu_count(), require="sharedmem")(
                    delayed(cache_label)(i) for i in range(total)
                )
        
        # NOTE: Check for any changes btw the cached labels
        if isinstance(cache_labels, dict):
            cache_labels = SortedDict(cache_labels)
        for (k, v) in cache_labels.items():
            if v.image_info.path != k:
                self.caching_labels = True
                break
        
        # NOTE: Write cache
        console.log(f"Labels has been cached to: {path}.")
        if isinstance(self.label_paths, str):
            cache_labels["hash"] = get_hash(
                [self.label_paths] + self.custom_label_paths + self.image_paths
            )
        else:
            cache_labels["hash"] = get_hash(
                self.label_paths + self.custom_label_paths + self.image_paths
            )
        torch.save(cache_labels, path)  # Save for next time
        return cache_labels
    
    @abstractmethod
    def load_label(
        self,
        image_path	     : str,
        label_path		 : str,
        custom_label_path: Optional[str] = None
    ) -> VisionData:
        """Load label data associated with the image from the corresponding
        label file.
        
        Args:
            image_path (str):
                Image file.
            label_path (str):
                Label file.
            custom_label_path (str, optional):
                Custom label file. Default: `None`.
                
        Returns:
            data (VisionData):
                `VisionData` object.
        """
        pass
    
    @abstractmethod
    def load_labels(
        self, image_paths: list[str], label_path: str,
    ) -> dict[str, VisionData]:
        """Load all labels from one label file.

        Args:
            image_paths (list[str]):
                List of image files.
            label_path (str):
                Label file.
        
        Returns:
            data (dict):
                Dictionary of `VisionData` objects.
        """
        pass
    
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        gb = 0  # Gigabytes of cached images
        with download_bar() as pbar:
            # Should be max 10k images
            for i in pbar.track(
                range(len(self.image_paths)), description="[red]Caching images"
            ):
                # image, hw_original, hw_resized
                (self.data[i].image,
                 self.data[i].image_info) = self.load_image(index=i)
                gb += self.data[i].image.nbytes
                # pbar.desc = "Caching images (%.1fGB)" % (gb / 1E9)
    
    def load_image(self, index: int) -> tuple[np.ndarray, ImageInfo]:
        """Load 1 image from dataset and preprocess image.
        
        Args:
            index (int):
                Index.
                
        Returns:
            image (np.ndarray):
                Image.
            info (ImageInfo):
                `ImageInfo` object.
        """
        image = self.data[index].image
        info  = self.data[index].image_info
        
        if image is None:  # Not cached
            path  = self.image_paths[index]
            image = read_image(path, backend=self.vision_backend)  # RGB
            if image is None:
                raise ValueError(f"Image not found at: {path}.")
            
            # NOTE: Resize image while keeping the image ratio
            """h0, w0 = image.shape[:2]  # Original HW
            ratio  = self.image_size / max(h0, w0)  # Resize image to image_size
            h1, w1 = int(h0 * ratio), int(w0 * ratio)
            if ratio != 1:  # Always resize down, only resize up if training with augmentation
                interpolation = cv2.INTER_AREA if (ratio < 1 and not self.augment) else cv2.INTER_LINEAR
                image         = cv2.resize(image, (w1, h1), interpolation=interpolation)"""
            h0, w0 = get_image_size(image)
            image  = resize(image, self.shape)
            h1, w1 = get_image_size(image)
            
            # NOTE: Assign image info if it has not been defined (just to be sure)
            info = ImageInfo.from_file(image_path=path, info=info)
            
            info.height = h1 if info.height != h1 else info.height
            info.width  = w1 if info.width  != w1 else info.width
            info.depth  = (image.shape[2] if info.depth != image.shape[2]
                           else info.depth)
        
        return image, info
    
    # MARK: Post-Load Data
    
    def post_load_data(self):
        """Post load data operations. We prepare `batch_shapes` for
        `rect_training` augmentation, and some labels statistics. If you want
        to add more operations, just `extend` this method.
        """
        # NOTE: Prepare for rectangular training
        if isinstance(self.load_augment, dict) and self.load_augment.get("rect", False):
            self.prepare_for_rect_training()
        
        # NOTE: Write data to our custom label format
        if not self.has_custom_labels and self.write_labels:
            self.write_custom_labels()

    # MARK: Get Item
    
    def get_item(self, index: int) -> tuple[Tensor, Tensor, Dim3]:
        """Get the item.
  
        Args:
            index (int):
                Index.
  
        Returns:
            input (Tensor):
                Image.
            target (Tensor):
                Bounding boxes.
            shape (Dim3):
                Shape of the resized images.
        """
        input = target = shape = None
        data  = self.data[index]
        
        # NOTE: Augmented load input
        if isinstance(self.load_augment, dict):
            mosaic = self.load_augment.get("mosaic", 0)
            mixup  = self.load_augment.get("mixup",  0)
            rect   = self.load_augment.get("rect",   False)
            if torch.rand(1) <= mosaic and not rect:  # Load mosaic
                input, target = self.load_mosaic(index)
                shape         = input.shape
                if torch.rand(1) <= mixup:  # Mixup
                    input, target = self.load_mixup(input, target)
        
        # NOTE: Load input normally
        if input is None:
            input , info = self.load_image(index=index)
            (h0, w0, _)  = info.shape0
            (h,  w,  _)  = info.shape

            # Letterbox
            if isinstance(self.load_augment, dict) and self.load_augment.get("rect", False):
                shape = self.batch_shapes[self.batch_indexes[index]]
            else:
                shape = self.image_size
            scale_up          = self.augment is not None
            input, ratio, pad = letterbox_resize(
                image=input, new_shape=shape, auto=False, scale_up=scale_up
            )
            shape = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            target = data.bbox_labels
            new_h  = ratio[1] * h
            new_w  = ratio[0] * w
            # Normalized xywh to xyxy format
            target[:, 2:6] = bbox_cxcywh_norm_to_xyxy(
                target[:, 2:6], new_h, new_w
            )
            # Add padding
            target[:, 2:6] = shift_bbox(target[:, 2:6], pad[1], pad[0])
        
        # NOTE: Augment
        input, target = self.augment(input=input, target=target)
        
        # NOTE: Convert to tensor
        input  = to_tensor(input, normalize=True).to(torch.uint8)
        target = torch.from_numpy(target)
        target = target.to(torch.get_default_dtype())
        
        return input, target, shape
    
    def load_mosaic(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load 4 images and create a mosaic.
        
        Args:
            index (int):
                Index.
                
        Returns:
            input4 (np.ndarray):
                Mosaic input.
            target4 (np.ndarray):
                Mosaic-ed bbox_labels.
        """
        target4       = []
        s             = self.image_size
        yc, xc        = s, s  # Mosaic center x, y
        mosaic_border = [-s // 2, -s // 2]
        # 3 additional input indices
        indices = [index] + \
                  [int(torch.randint(len(self.data) - 1, (1,))) for _ in range(3)]
        
        for i, index in enumerate(indices):
            # Load input
            input, info = self.load_image(index=index)
            h, w, _     = info.shape
            
            # Place input in input4
            if i == 0:  # Top left
                input4 = np.full((s * 2, s * 2, input.shape[2]), 114, np.uint8)
                # base input with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                # xmin, ymin, xmax, ymax (large input)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                # xmin, ymin, xmax, ymax (small input)
            elif i == 1:  # Top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # Bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # Bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            input4[y1a:y2a, x1a:x2a] = input[y1b:y2b, x1b:x2b]
            # input4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            
            # Labels
            target = self.data[index].bbox_labels
            if target.size > 0:
                target[:, 2:6] = bbox_cxcywh_norm_to_xyxy(target[:, 2:6], h, w)
                # Normalized xywh to pixel xyxy format
                target[:, 2:6] = shift_bbox(target[:, 2:6], padh, padw)
                # Add padding
            target4.append(target)
            
        # Concat/clip target
        if len(target4):
            target4 = np.concatenate(target4, 0)
            # np.clip(target4[:, 1:] - s / 2, 0, s, out=target4[:, 1:])
            # use with center crop
            np.clip(target4[:, 2:6], 0, 2 * s, out=target4[:, 2:6])
            # use with random_affine
    
            # Replicate
            # input4, target4 = replicate(input4, target4)
        
        return input4, target4
    
    def load_mixup(
        self, input: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """MixUp https://arxiv.org/pdf/1710.09412.pdf."""
        input2, target2 = self.load_mosaic(
            index=int(torch.randint(len(self.data) - 1, (1,)))
        )
        ratio  = np.random.beta(8.0, 8.0)
        # mixup ratio, alpha=beta=8.0
        input  = input * ratio + input2 * (1 - ratio)
        input  = input.astype(np.uint8)
        target = np.concatenate((target, target2), 0)
        return input, target

    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, Tensor, Dim3]:
        """Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the DataLoader wrapper.
        """
        input, target, shapes = zip(*batch)  # transposed
        for i, l in enumerate(target):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(input, 0), torch.cat(target, 0), shapes
    
    # MARK: Utils
    
    def prepare_for_rect_training(self):
        """Prepare `batch_shapes` for `rect_training` augmentation.
        
        References:
            https://github.com/ultralytics/yolov3/issues/232
        """
        rect   = self.load_augment.get("rect",   False)
        stride = self.load_augment.get("stride", 32)
        pad    = self.load_augment.get("pad",    0)
        
        if rect:
            # NOTE: Get number of batches
            n  = len(self.data)
            bi = np.floor(np.arange(n) / self.batch_size).astype(np.int)  # Batch index
            nb = bi[-1] + 1  # Number of batches
            
            # NOTE: Sort data by aspect ratio
            s     = [data.image_info.shape0 for data in self.data]
            s     = np.array(s, dtype=np.float64)
            ar    = s[:, 1] / s[:, 0]  # Aspect ratio
            irect = ar.argsort()
            
            self.image_paths = [self.image_paths[i] for i in irect]
            self.label_paths = [self.label_paths[i] for i in irect]
            self.data        = [self.data[i]        for i in irect]
            ar				 = ar[irect]
            
            # NOTE: Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            
            self.batch_shapes  = \
                stride * np.ceil(np.array(shapes)
                                 * self.image_size / stride
                                 + pad).astype(np.int)
            self.batch_indexes = bi
        
    def write_custom_labels(self):
        """Write all data to custom label files using our custom label format.
        """
        # NOTE: Get label files
        """
        parents     = [str(Path(file).parent) for file in self.label_paths]
        stems       = [str(Path(file).stem)   for file in self.label_paths]
        stems       = [stem.replace("_custom", "") for stem in stems]
        label_paths = [os.path.join(parent, f"{stem}_custom.json")
                       for (parent, stem) in zip(parents, stems)]
        """
        dirnames = [os.path.dirname(p) for p in self.custom_label_paths]
        create_dirs(paths=dirnames)
        
        # NOTE: Scan all images to get information
        with progress_bar() as pbar:
            for i in pbar.track(
                range(len(self.data)), description="Scanning images"
            ):
                # image, hw_original, hw_resized
                _, self.data[i].image_info = self.load_image(index=i)
            
            # NOTE: Parallel write labels
            for (data, path) in pbar.track(
                zip(self.data, self.custom_label_paths),
                description="Writing custom annotations",
                total=len(self.data)
            ):
                VisionDataHandler().dump_to_file(data=data, path=path)
