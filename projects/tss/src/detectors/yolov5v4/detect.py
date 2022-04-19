import argparse
import time
from pathlib import Path
import numpy as np
import pickle
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, \
	xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import sys


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
	# Rescale coords (xyxy) from img1_shape to img0_shape
	if ratio_pad is None:  # calculate from img0_shape
		gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
		pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
	else:
		gain = ratio_pad[0][0]
		pad = ratio_pad[1]

	coords[:, [0, 2]] -= pad[0]  # x padding
	coords[:, [1, 3]] -= pad[1]  # y padding
	coords[:, :4] /= gain
	# clip_coords(coords, img0_shape)
	return coords

def detect(save_img=False):
	out, source, weights, view_img, save_txt, imgsz, name_video, roi_path =  \
		opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.name, opt.roi_region
	webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
		('rtsp://', 'rtmp://', 'http://'))

	# Get the ignore region
	ignore_region = cv2.imread(roi_path)

	save_dir = Path(out)
	os.makedirs(os.path.join(save_dir, 'dets_debug', 'yolov5x'), exist_ok=True)
	os.makedirs(os.path.join(save_dir, 'dets_label', 'yolov5x', name_video), exist_ok=True)
	os.makedirs(os.path.join(save_dir, 'dets_crop', 'yolov5x', name_video), exist_ok=True)
	os.makedirs(os.path.join(save_dir, 'dets_crop_pkl', 'yolov5x', name_video), exist_ok=True)

	# Initialize
	set_logging()
	device = select_device(opt.device)
	half = device.type != 'cpu'  # half precision only supported on CUDA

	# Load model
	model = attempt_load(weights, map_location=device)  # load FP32 model
	stride = int(model.stride.max())  # model stride
	imgsz = check_img_size(imgsz, s=stride)  # check img_size
	if half:
		model.half()  # to FP16

	# Second-stage classifier
	classify = False
	if classify:
		modelc = load_classifier(name='resnet101', n=2)  # initialize
		modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

	# Set Dataloader
	vid_path, vid_writer = None, None

	# Create dataloader
	save_img = True
	dataset = LoadImages(source, img_size=imgsz, stride=stride, ignore_region=ignore_region)

	out_dict=dict()

	# Get names and colors
	names = model.module.names if hasattr(model, 'module') else model.names
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

	# Run inference
	if device.type != 'cpu':
		model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
	t0 = time.time()

	for index_image, (path, img, im0s, vid_cap) in enumerate(dataset):
		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		# Inference
		t1 = time_synchronized()
		pred = model(img, augment=opt.augment)[0]

		# Apply NMS
		pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
		t2 = time_synchronized()

		# Apply Classifier
		if classify:
			pred = apply_classifier(pred, modelc, img, im0s)

		# Process detections
		for i, det in enumerate(pred):  # detections per image
			if webcam:  # batch_size >= 1
				p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
			else:
				p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

			p = Path(p)  # to Path
			save_path = os.path.join(save_dir, 'dets_debug', 'yolov5x', f"{name_video}.mp4")
			txt_path = os.path.join(save_dir, 'dets_label', 'yolov5x', name_video, f"{index_image:08d}")
			det_path = os.path.join(save_dir, 'dets_crop', 'yolov5x', name_video, f"{index_image:08d}")

			s += '%gx%g ' % img.shape[2:]  # print string
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
			if len(det):
				img_det = np.copy(im0)
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
				det_num = 0
				# Write results
				for *xyxy, conf, cls in reversed(det):
					x1, y1, x2, y2 = tuple(torch.tensor(xyxy).view(4).tolist())
					x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

					if x1 < 0 or y1 < 0 or x2 > im0.shape[1]-1 or y2 > im0.shape[0]-1:
						continue

					if save_txt:  # Write to file
						xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
						line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
						with open(txt_path + '.txt', 'a') as f:
							f.write(('%g ' * len(line)).rstrip() % line + '\n')

					if save_img or view_img:  # Add bbox to image
						label = f'{names[int(cls)]} {conf:.2f}'
						plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

					if True:
						det_name = f"{index_image:08d}" + "_{:0>6d}".format(det_num)
						det_img_path = det_path + "_{:0>6d}.jpg".format(det_num)
						det_class = int(cls.tolist())
						det_conf = conf.tolist()
						cv2.imwrite(det_img_path, img_det[y1:y2, x1:x2])
						out_dict[det_name] = {
							'bbox'   : (x1, y1, x2, y2),
							'frame'  : f"{index_image:08d}",
							'id'     : det_num,
							'imgname': det_name + ".jpg",
							'class'  : det_class,
							'conf'   : det_conf
						}

					det_num+=1

			# Print time (inference + NMS)
			print(f'{s}Done. ({t2 - t1:.3f}s)')

			# Stream results
			if view_img:
				cv2.imshow(str(p), im0)
				if cv2.waitKey(1) == ord('q'):  # q to quit
					raise StopIteration

			# Save results (image with detections)
			if save_img:
				if vid_path != save_path:  # new video
					vid_path = save_path
					if isinstance(vid_writer, cv2.VideoWriter):
						vid_writer.release()  # release previous video writer

					fourcc = 'mp4v'  # output video codec
					if vid_cap is not None:
						fps = vid_cap.get(cv2.CAP_PROP_FPS)
						w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
						h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
					else:
						fps = 10
						h, w, _ = im0.shape

					vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
				vid_writer.write(im0)

		# NOTE: save pickle
		pickle.dump(out_dict, open(
			os.path.join(save_dir, 'dets_crop_pkl', 'yolov5x', name_video, f'{name_video}_dets_crop.pkl'), 'wb'))

	print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
	parser.add_argument('--name', type=str, default='c041', help='Name of video')  # file/folder, 0 for webcam
	parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
	parser.add_argument('--roi_region', type=str, default='inference/roi.jpg', help='ignor region')  # output folder
	parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')
	parser.add_argument('--update', action='store_true', help='update all models')
	parser.add_argument('--project', default='runs/detect', help='save results to project/name')
	parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
	opt = parser.parse_args()
	print(opt)

	with torch.no_grad():
		if opt.update:  # update all models (to fix SourceChangeWarning)
			for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
				detect()
				strip_optimizer(opt.weights)
		else:
			detect()
