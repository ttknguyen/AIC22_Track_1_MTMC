#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
from typing import Callable

import torch
from torch import distributed as dist

from torchkit.core.utils import console

__all__ = [
    "get_dist_info",
    "get_next_version",
    "named_apply"
]


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank       = 0
        world_size = 1
    return rank, world_size


# noinspection PyTypeChecker
def get_next_version(root_dir: str) -> int:
    """Get the next experiment version number.
    
    Args:
        root_dir (str):
            Path to the folder that contains all experiment folders.

    Returns:
        version (int):
            Next version number.
    """
    try:
        listdir_info = os.listdir(root_dir)
    except OSError:
        # console.log(f"Missing folder: {root_dir}")
        return 0
    
    existing_versions = []
    for listing in listdir_info:
        if isinstance(listing, str):
            d = listing
        else:
            d = listing["name"]
        bn = os.path.basename(d)
        if bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace("/", "")
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0
    
    return max(existing_versions) + 1


def named_apply(
    fn          : Callable,
    module      : torch.nn.Module,
    name        : str       = "",
    depth_first : bool      = True,
    include_root: bool      = False
) -> torch.nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn, module=child_module, name=child_name,
            depth_first=depth_first, include_root=True
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module
