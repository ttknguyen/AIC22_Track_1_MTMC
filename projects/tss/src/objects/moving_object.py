#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Newly detected object from detector model. Attributes includes:
bounding box, confident score, class, uuid, ...
"""

from __future__ import annotations

import abc
import enum
from typing import Union

import cv2
import numpy as np

from torchkit.core.math import euclidean_distance
from torchkit.core.vision import AppleRGB
from projects.tss.src.camera.roi import ROI
from projects.tss.src.trackers.motion import Motion
from .base import BaseObject
from .instance import Instance

__all__ = [
    "MovingObject",
    "MovingState"
]


# MARK: - MovingState

class MovingState(enum.Enum):
    """An enum that identify the counting state of an object when moving
    through the camera.
    """
    Candidate   = 1  # Preliminary state.
    Confirmed   = 2  # Confirmed the Detection is a road_objects eligible for
                     # counting.
    Counting    = 3  # Object is in the counting zone/counting state.
    ToBeCounted = 4  # Mark object to be counted somewhere in this loop
                     # iteration.
    Counted     = 5  # Mark object has been counted.
    Exiting     = 6  # Mark object for exiting the ROI or image frame.
                     # Let's it die by itself.

    @staticmethod
    def values() -> list[int]:
        """Return the list of all values."""
        return [s.value for s in MovingState]

    @staticmethod
    def keys():
        """Return the list of all enum keys."""
        return [s for s in MovingState]


# MARK: - MovingObject

class MovingObject(BaseObject, metaclass=abc.ABCMeta):
    """Moving Object.

    Attributes:
        motion (Motion):
            Motion model.
        moving_state (MovingState):
            Current state of the moving object with respect to camera's
            ROIs. Default: `Candidate`.
        moi_id (int, str, optional):
            Id of the MOI that the current moving object is best fitted to.
            Default: `None`.
        trajectory (np.ndarray):
            Object trajectory as an array of instances' center points.
    """

    # MARK: Class Attributes

    min_entering_distance: float = 0.0    # Min distance when an object enters the ROI to be `Confirmed`. Default: `0.0`.
    min_traveled_distance: float = 100.0  # Min distance between first trajectory point with last trajectory point. Default: `10.0`.
    min_hit_streak       : int   = 10     # Min number of `consecutive` frame has that track appear. Default: `10`.
    max_age              : int   = 1      # Max frame to wait until a dead track can be counted. Default: `1`.

    # MARK: Magic Functions

    def __init__(
        self,
        motion      : Motion,
        moving_state: MovingState           = MovingState.Candidate,
        moi_id      : Union[int, str, None] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.motion       = motion
        self.moving_state = moving_state
        self.moi_id       = moi_id
        self.trajectory   = np.array([self.current_bbox_center])

    # MARK: Properties

    @property
    def traveled_distance(self) -> float:
        """Return the traveled distance of the object."""
        if len(self.trajectory) < 2:
            return 0.0
        return euclidean_distance(self.trajectory[0], self.trajectory[-1])

    @property
    def hits(self) -> int:
        """Return the number of frame has that track appear."""
        return self.motion.hits

    @property
    def hit_streak(self) -> int:
        """Return the number of `consecutive` frame has that track appear."""
        return self.motion.hit_streak

    @property
    def age(self) -> int:
        """Return the number of frame while the track is alive,
        from Candidate -> Deleted."""
        return self.motion.age

    @property
    def time_since_update(self) -> int:
        """Return the number of `consecutive` frame that track disappear."""
        return self.motion.time_since_update

    # MARK: Properties (MovingState)

    @property
    def moving_state(self) -> MovingState:
        """Return moving state."""
        return self._moving_state

    @moving_state.setter
    def moving_state(self, moving_state: MovingState):
        """Assign moving state.

        Args:
            moving_state (MovingState):
                Object's moving state.
        """
        if moving_state not in MovingState.keys():
            raise ValueError(f"Moving state should be one of: "
                             f"{MovingState.keys()}. But given {moving_state}.")
        self._moving_state = moving_state

    @property
    def is_candidate(self) -> bool:
        """Return `True` if the current moving state is `Candidate`."""
        return self.moving_state == MovingState.Candidate

    @property
    def is_confirmed(self) -> bool:
        """Return `True` if the current moving state is `Confirmed`."""
        return self.moving_state == MovingState.Confirmed

    @property
    def is_counting(self) -> bool:
        """Return `True` if the current moving state is `Counting`."""
        return self.moving_state == MovingState.Counting

    @property
    def is_countable(self) -> bool:
        """Return `True` if the current vehicle is countable."""
        return True if (self.moi_id is not None) else False

    @property
    def is_to_be_counted(self) -> bool:
        """Return `True` if the current moving state is `ToBeCounted`."""
        return self.moving_state == MovingState.ToBeCounted

    @property
    def is_counted(self) -> bool:
        """Return `True` if the current moving state is `Counted`."""
        return self.moving_state == MovingState.Counted

    @property
    def is_exiting(self) -> bool:
        """Return `True` if the current moving state is `Exiting`."""
        return self.moving_state == MovingState.Exiting

    # MARK: Update

    def update(self, instance: Instance, **kwargs):
        """Update with value from a `Instance` object.

        Args:
            instance (Instance):
                Instance of the object.
        """
        super(MovingObject, self).update(instance=instance, **kwargs)
        self.motion.update_motion_state(instance=instance)
        self.update_trajectory()

    def update_trajectory(self):
        """Update trajectory with instance's center point."""
        traveled_distance = euclidean_distance(
            self.trajectory[-1], self.current_bbox_center
        )
        if traveled_distance >= self.min_traveled_distance:
            self.trajectory = np.append(
                self.trajectory, [self.current_bbox_center], axis=0
            )

    def update_moving_state(self, rois: list[ROI], **kwargs):
        """Update the current state of the road_objects. One recommendation of the the state diagram is as follow:

                (exist >= 10 frames)  (road_objects cross counting line)   (after being counted
                (in roi)                                               by counter)
        _____________          _____________                  ____________        ___________        ________
        | Candidate | -------> | Confirmed | ---------------> | Counting | -----> | Counted | -----> | Exit |
        -------------          -------------                  ------------        -----------        --------
              |                       |                                                                  ^
              |_______________________|__________________________________________________________________|
                                (mark by tracker when road_objects's max age > threshold)
        """
        roi = next(
            (roi for roi in rois if roi.id_ == self.current_roi_id), None
        )
        if roi is None:
            return
        
        # NOTE: From Candidate --> Confirmed
        if self.is_candidate:
            entering_distance = roi.is_bbox_in_or_touch_roi(
                bbox_xyxy=self.current_bbox, compute_distance=True
            )
            if (
                self.hit_streak >= MovingObject.min_hit_streak and
                entering_distance >= MovingObject.min_entering_distance and
                self.traveled_distance >= MovingObject.min_traveled_distance
            ):
                self.moving_state = MovingState.Confirmed

        # NOTE: From Confirmed --> Counting
        elif self.is_confirmed:
            if roi.is_bbox_in_or_touch_roi(bbox_xyxy=self.current_bbox) <= 0:
                self.moving_state = MovingState.Counting

        # NOTE: From Counting --> ToBeCounted
        elif self.is_counting:
            if (
                roi.is_center_in_or_touch_roi(bbox_xyxy=self.current_bbox) < 0 or
                self.time_since_update >= self.max_age
            ):
                self.moving_state = MovingState.ToBeCounted

        # NOTE: From ToBeCounted --> Counted
        # Perform when counting the vehicle

        # NOTE: From Counted --> Exiting
        elif self.is_counted:
            if (
                roi.is_center_in_or_touch_roi(
                    bbox_xyxy=self.current_bbox, compute_distance=True
                ) <= 0 or
                self.time_since_update >= MovingObject.max_age
            ):
                self.moving_state = MovingState.Exiting

    # MARK: Visualize

    def draw(self, drawing: np.ndarray, **kwargs):
        """Draw the object into the `drawing`.

        Args:
            drawing (np.ndarray):
                Drawing canvas.
        """
        if self.moi_id is not None:
            color = AppleRGB.values()[self.moi_id]
        else:
            color = self.label_by_majority["color"]
            
        if self.is_confirmed:
            BaseObject.draw(self, drawing=drawing, label=False, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_counting:
            BaseObject.draw(self, drawing=drawing, label=True, **kwargs)
        elif self.is_counted:
            BaseObject.draw(
                self, drawing=drawing, label=True, color=color, **kwargs
            )
        elif self.is_exiting:
            BaseObject.draw(
                self, drawing=drawing, label=True, color=color, **kwargs
            )

    def draw_trajectory(self, drawing: np.ndarray):
        if self.moi_id is not None:
            color = AppleRGB.values()[self.moi_id]
        else:
            color = self.label_by_majority["color"]
            
        if self.trajectory is not None:
            pts = self.trajectory.reshape((-1, 1, 2))
            cv2.polylines(
                img       = drawing,
                pts       = [pts.astype(int)],
                isClosed  = False,
                color     = color,
                thickness = 2
            )
            for point in self.trajectory:
                cv2.circle(
                    img       = drawing,
                    center    = tuple(point.astype(int)),
                    radius    = 3,
                    thickness = 2,
                    color     = color
                )
