#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""IO classes especially for AI City Challenge tasks.
"""

from __future__ import annotations

import os
from operator import itemgetter
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional

from torchkit.core.file import create_dirs
from torchkit.core.file import is_stem
from projects.tss.src.objects import MovingObject
from projects.tss.utils import data_dir

__all__ = [
	"AICCountingWriter",
	"AICTrackingWriter",
	"video_map"
]


# MARK: - Global

# Numeric identifier of input camera stream
video_map = {
	"cam_1"     : 1,
	"cam_1_dawn": 2,
	"cam_1_rain": 3,
	"cam_2"     : 4,
	"cam_2_rain": 5,
	"cam_3"     : 6,
	"cam_3_rain": 7,
	"cam_4"     : 8,
	"cam_4_dawn": 9,
	"cam_4_rain": 10,
	"cam_5"     : 11,
	"cam_5_dawn": 12,
	"cam_5_rain": 13,
	"cam_6"     : 14,
	"cam_6_snow": 15,
	"cam_7"     : 16,
	"cam_7_dawn": 17,
	"cam_7_rain": 18,
	"cam_8"     : 19,
	"cam_9"     : 20,
	"cam_10"    : 21,
	"cam_11"    : 22,
	"cam_12"    : 23,
	"cam_13"    : 24,
	"cam_14"    : 25,
	"cam_15"    : 26,
	"cam_16"    : 27,
	"cam_17"    : 28,
	"cam_18"    : 29,
	"cam_19"    : 30,
	"cam_20"    : 31
}


# MARK: - AICCountingWriter

# noinspection PyAttributeOutsideInit
class AICCountingWriter:
	"""AIC Counting Writer periodically saves the counting results in ONE
	camera.
	
	Attributes:
		dst (str):
			Path to the file to save the counting results.
		camera_name (str):
			Camera name.
		video_id (int):
			Numeric identifier of input camera stream.
		start_time (float):
			Moment when the TexIO is initialized.
		writer (io stream):
			File writer to export the counting results.
	"""

	# MARK: Magic Function

	def __init__(
		self,
		dst 	   : str,
		camera_name: str,
		start_time : float = timer(),
		*args, **kwargs
	):
		super().__init__()
		if camera_name not in video_map:
			raise ValueError(
				f"The given `camera_name` has not been defined in AIC camera "
			    f"list. Please check again!"
			)

		self.dst		 = dst
		self.camera_name = camera_name
		self.video_id 	 = video_map[camera_name]
		self.start_time  = start_time
		self.lines 		 = []
		# self.init_writer(dst=self.dst)
		
	def __del__(self):
		""" Close the writer object."""
		"""
		if self.writer:
			self.writer.close()
		"""
		pass

	# MARK: Configure:

	def init_writer(self, dst: str):
		"""Initialize writer object.

		Args:
			dst (str):
				Path to the file to save the counting results.
		"""
		if is_stem(dst):
			dst = f"{dst}.txt"
		elif os.path.isdir(dst):
			dst = os.path.join(dst, f"{self.camera_name}.txt")
		parent_dir = str(Path(dst).parent)
		create_dirs(paths=[parent_dir])
		# self.writer = open(dst, "w")

	# MARK: Write
	
	def write(self, moving_objects: list[MovingObject]):
		"""Write counting result from a list of tracked moving objects.
		
		Each line in the output format of AI City Challenge contains:
		<gen_time> <video_id> <frame_id> <MOI_id> <vehicle_class_id>
			<gen_time>:         [float] is the generation time until this
										frame’s output is generated, from the
										start of the program execution, in
										seconds.
			<video_id>:         [int] is the video numeric identifier, starting
									  with 1.
			<frame_id>:         [int] represents the frame count for the
									  current frame in the current video,
									  starting with 1.
			<MOI_id>:           [int] denotes the the movement numeric
									  identifier of the movement of that video.
			<vehicle_class_id>: [int] is the road_objects classic numeric
									  identifier, where 1 stands for “car”
									  and 2 represents “truck”.
		
		Args:
			moving_objects (list):
				List of tracked moving objects.
		"""
		for obj in moving_objects:
			gen_time = format((obj.timestamp - self.start_time), ".2f")
			frame_id = obj.current_frame_index
			moi_id   = obj.moi_id
			class_id = obj.label_id_by_majority
			if class_id in [1, 2]:
				line = (f"{gen_time} {self.video_id} {frame_id} {moi_id} "
					    f"{class_id}\n")
				self.lines.append(line)
				# self.writer.write(line)
	
	def dump(self):
		dst = self.dst
		if is_stem(dst):
			dst = f"{dst}.txt"
		elif os.path.isdir(dst):
			dst = os.path.join(dst, f"{self.camera_name}.txt")
		parent_dir = str(Path(dst).parent)
		create_dirs(paths=[parent_dir])
		
		with open(dst, "w") as f:
			for line in self.lines:
				f.write(line)
		

# MARK: - AICTrackingWriter

class AICTrackingWriter:
	pass


# MARK: - Compress result

def compress_all_result(
	output_dir: Optional[str] = None, output_name: Optional[str] = None
):
	""" Compress all result of video into one file
	
	Args:
		output_dir (str):
			Directory of output track1.txt will be written
		output_name:
			Final compress result name
				e.g.: "track1.txt"
	"""
	output_dir 		= output_dir if (output_dir is not None) else data_dir
	output_name 	= output_name if (output_name is not None) else "track1"
	output_name 	= os.path.join(output_dir, f"{output_name}.txt")
	compress_writer = open(output_name, "w")

	# NOTE: Get result from each file
	for video_name, video_id in video_map.items():
		video_result_path = os.path.join(output_dir, f"{video_name}.txt")

		if not os.path.exists(video_result_path):
			print(f"Result of {video_result_path} is not exist")
			continue

		# NOTE: Read result
		results = []
		with open(video_result_path) as f:
			line = f.readline()
			while line:
				words  = line.split(' ')
				result = {
					"gen_time"   : float(words[0]),
					"video_id"   : int(words[1]),
					"frame_id"   : int(words[2]),
					"movement_id": int(words[3]),
					"class_id"   : int(words[4]),
				}
				if result["class_id"] != 0:
					results.append(result)
				line = f.readline()

		# TODO: Sort result
		results = sorted(
			results, key=itemgetter("gen_time", "frame_id", "movement_id")
		)

		# TODO: write result
		for result in results:
			compress_writer.write(f"{result['gen_time']} ")
			compress_writer.write(f"{result['video_id']} ")
			compress_writer.write(f"{result['frame_id']} ")
			compress_writer.write(f"{result['movement_id']} ")
			compress_writer.write(f"{result['class_id']}")
			compress_writer.write("\n")

	compress_writer.close()
