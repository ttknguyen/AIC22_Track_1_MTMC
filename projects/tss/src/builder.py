#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory classes.
"""

from __future__ import annotations

from torchkit.core import Factory

__all__ = [
	"CAMERAS",
	"DETECTORS",
	"REIDENTIFIER",
	"MOTIONS",
	"TRACKERS"
]

CAMERAS      = Factory(name="cameras")
DETECTORS    = Factory(name="object_detectors")
REIDENTIFIER = Factory(name="reidentifiers")
MOTIONS      = Factory(name="motions")
TRACKERS     = Factory(name="trackers")
