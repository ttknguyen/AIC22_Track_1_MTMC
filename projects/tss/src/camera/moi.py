#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Movement of interest.
"""

from __future__ import annotations

import os
from typing import Optional
from typing import Union

import numpy as np
from cv2 import cv2

from torchkit.core import DISTANCES
from torchkit.core.file import is_json_file
from torchkit.core.math import angle_between_vectors
from torchkit.core.type import Color
from torchkit.core.vision import AppleRGB
from projects.tss.utils import data_dir
from projects.tss.utils import load_config

__all__ = [
	"MOI"
]


# MARK: - MOI (Movement of Interest)

class MOI:
	"""MOI (Movement of Interest)
	
	Attributes:
		id_ (int, str, optional):
			MOI's unique id. Default: `None`.
		shape_type (str, optional):
			Shape type. Default: `None`.
		offset (int):
			Default: `None`.
		distance_function (callable):
			Distance function. Default: `hausdorff`.
		distance_threshold (float):
			Maximum distance for counting with track. Default: `300.0`.
		angle_threshold (float):
			Maximum angle for counting with track. Default: `45.0`.
		color (Color, optional):
			Default: `None`.
	"""

	# MARK: Magic Functions

	def __init__(
		self,
		# From `rmois.json` file.
		id_               : Union[int, str, None]          	  = None,
		points            : Optional[Union[np.ndarray, list]] = None,
		shape_type        : Optional[str]                     = None,
		offset            : Optional[int]                 	  = None,
		# From `config.yaml` file.
		distance_function : str                           	  = "hausdorff",
		distance_threshold: float                         	  = 300.0,
		angle_threshold   : float                         	  = 45.0,
		color             : Color          	 				  = None,
		*args, **kwargs
	):
		"""
		
		Args:
			id_ (int, str, optional):
				MOI's unique id. Default: `None`.
			points (np.ndarray, list, optional):
				List of points in the MOI.
			shape_type (str, optional):
				Shape type. Default: `None`.
			offset (int):
				Default: `None`.
			distance_function (str):
				Distance function. Default: `hausdorff`.
			distance_threshold (float):
				Maximum distance for counting with track. Default: `300.0`.
			angle_threshold (float):
				Maximum angle for counting with track. Default: `45.0`.
			color (Color, optional):
				Default: `None`.
		"""
		super().__init__()
		self.id_                = id_
		self.points             = points
		self.shape_type         = shape_type
		self.offset             = offset
		self.distance_threshold = distance_threshold
		self.angle_threshold    = angle_threshold
		
		if (id_ is None) or isinstance(id_, str):
			self.color = AppleRGB.WHITE.value
		else:
			self.color = color if color else AppleRGB.values()[id_]
		
		self.distance_function = DISTANCES.build(name=distance_function)

	# MARK: Properties
	
	@property
	def points(self) -> Union[np.ndarray, None]:
		"""Returns the array of points in the MOI."""
		return self._points
	
	@points.setter
	def points(self, points: Union[np.ndarray, list, None]):
		"""Assign points in MOI.

		Args:
			points (np.ndarray, list, optional):
				List of points in the MOI. Default: `None`.
		"""
		if isinstance(points, list):
			points = np.array(points, np.int32)
		self._points = points

	@property
	def are_valid_points(self) -> bool:
		"""Returns `True` if the points are valid."""
		if self.points and not all(len(track) >= 2 for track in self.points):
			return True
		else:
			print(f"Insufficient number of points in the MOI's track.")
			return False

	# MARK: Configure
	
	@classmethod
	def load_from_file(
		cls, file: str, dataset: Optional[str] = None, **kwargs
	) -> list:
		"""Load MOIs' points from external `.json` file.
		
		Args:
			file (string):
				Give the roi file. Example a path
				"..data/aicity2021/roi_data/cam_n.json", so provides
				`cam_n.json`.
			dataset (string, optional):
				Name of the dataset to work on. Default: `None`.
		
		Returns:
			mois (list):
				Return the list of MOIs.
		"""
		# NOTE: Get json file
		if dataset:
			path = os.path.join(data_dir, dataset, "rmois", file)
		else:
			path = os.path.join(data_dir, "rmois", file)
		assert is_json_file(path=path), (f"File not found or given a wrong "
										 f"file type at {path}.")

		# NOTE: Create MOIs
		mois_data = load_config(path).moi
		mois	  = []
		for moi_cfg in mois_data:
			mois.append(cls(**moi_cfg, **kwargs))
		return mois
	
	# MARK: Matching
	
	@staticmethod
	def associate_moving_objects_to_mois(
		objs: list, mois: list, shape_type: str = "linestrip"
	):
		"""Static method to check if a list of given moving objects belong
		to one of the MOIs in the image.

		Args:
			objs (list):
				List of objects.
			mois (list):
				List of MOIs in the image.
			shape_type (str):
				Shape of MOI to check. One of: ["polygon", "linestrip"].
				Default: `linestrip`.
		"""
		if len(objs) <= 0:
			return
		polygon_mois   = [m for m in mois if m.shape_type == "polygon"]
		linestrip_mois = [m for m in mois if m.shape_type == "linestrip"]
		
		if shape_type == "polygon":
			for o in objs:
				if o.moi_id is None:
					o.moi_id = MOI.find_moi_for_bbox(bbox_xyxy=o.current_bbox,
													 mois=polygon_mois)
		elif shape_type == "linestrip":
			for o in objs:
				if o.moi_id is None:
					o.moi_id = MOI.best_matched_moi(object_track=o.trajectory,
													mois=linestrip_mois)[0]
		
	@staticmethod
	def find_moi_for_bbox(
		bbox_xyxy: np.ndarray, mois: np.ndarray
	) -> Optional[int]:
		"""Static method to check if a given bbox belong to one of the many
		MOIs in the image.

		Args:
			bbox_xyxy (np.ndarray):
				Bbox's coordinates as [x, y, x, y].
			mois (np.ndarray):
				Array of MOIs.
		
		Returns:
			moi_id (int, optional):
				MOI's id_ that the object is in. Else None.
		"""
		for moi in mois:
			if moi.is_center_in_or_touch_moi(bbox_xyxy=bbox_xyxy) > 0:
				return moi.id_
		return None
	
	@staticmethod
	def best_matched_moi(
		object_track: np.ndarray, mois: list
	) -> tuple[int, float]:
		"""Find the Moi that best matched with the object's track.
		
		Args:
			object_track (np.ndarray):
				Object's track as an array of points.
			mois (np.ndarray):
				Array of MOIs.

		Returns:
			min_moi_id (id_):
				Best match MOI's id_.
			min_distance (float):
				Best match MOI's min distance.
		"""
		# NOTE: Calculate distances between object track and all mois' tracks
		distances = []
		angles    = []
		for moi in mois:
			distances.append(
				moi.distances_with_track(object_track=object_track)
			)
			angles.append(
				moi.angles_with_track(object_track=object_track)
			)
		
		min_moi_id, min_distance = None, None
		for i, (d, a) in enumerate(zip(distances, angles)):
			if (d is None) or (a is None):
				continue
			if (min_distance is not None) and (min_distance < d):
				continue
			min_distance = d
			min_moi_id   = mois[i].id_

		return min_moi_id, min_distance

	def distance_with_track(self, object_track: np.ndarray) -> float:
		"""Calculate the distance between object's track to the MOI's tracks.
		If `distance > self.distance_threshold`, return `None`.
		
		Args:
			object_track (np.ndarray):
				Object's trajectory as an array of points.
				
		Returns:
			distance (float):
				Distance value between object's track with MOI's track.
		"""
		distance = self.distance_function(self.points, object_track)
		return None if (distance > self.distance_threshold) else distance
	
	def angle_with_track(self, object_track: np.ndarray) -> float:
		"""Calculate the angle between object's track to the MOI's tracks.
		If `angle > self.angle_threshold`, return `None`.
		
		Args:
			object_track (np.ndarray):
				Object's trajectory as an array of points.
				
		Returns:
			angle (float):
				Angle value between object's track with MOI's track.
		"""
		angle = angle_between_vectors(self.points, object_track)
		return None if (angle > self.angle_threshold) else angle
	
	def is_center_in_or_touch_moi(
		self, bbox_xyxy: np.ndarray, compute_distance: bool = False
	) -> int:
		"""Check the bounding box touch MOI or not.
		
		Args:
			bbox_xyxy (np.ndarray):
				Bbox coordinates as [x, y, x, y].
			compute_distance (bool):
				Should calculate the distance from center to moi?
				Default: `False`.
		
		Returns:
			(int):
				positive (inside), negative (outside), or zero (on an edge).
		"""
		c_x = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
		c_y = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
		return int(cv2.pointPolygonTest(
			self.points, (c_x, c_y), compute_distance
		))
	
	# MARK: Visualize
	
	def draw(self, drawing: np.ndarray):
		"""Draw the MOI.
		
		Args:
			drawing (np.ndarray):
				Drawing canvas.
		
		Returns:
			drawing (np.ndarray):
				Drawing canvas.
		"""
		# NOTE: Draw MOI's direction
		pts = self.points.reshape((-1, 1, 2))
		if self.shape_type == "polygon":
			cv2.polylines(
				img       = drawing,
				pts       = [pts],
				isClosed  = True,
				color     = self.color,
				thickness = 1,
				lineType  = cv2.LINE_AA
			)
		elif self.shape_type == "linestrip":
			cv2.polylines(
				img       = drawing,
				pts       = [pts],
				isClosed  = False,
				color     = self.color,
				thickness = 1,
				lineType  = cv2.LINE_AA
			)
			cv2.arrowedLine(
				img       = drawing,
				pt1       = tuple(self.points[-2]),
				pt2       = tuple(self.points[-1]),
				color     = self.color,
				thickness = 1,
				line_type = cv2.LINE_AA,
				tipLength = 0.2
			)
			for i in range(len(self.points) - 1):
				cv2.circle(
					img       = drawing,
					center    = tuple(self.points[i]),
					radius    = 3,
					color     = self.color,
					thickness = -1,
					lineType  = cv2.LINE_AA
				)
				
		# NOTE: Draw MOI's id_
		cv2.putText(
			img       = drawing,
			text      = f"{self.id_}",
			fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
			fontScale = 0.75,
			org       = tuple(self.points[-1]),
			color     = self.color,
			thickness = 2
		)
