
import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from Load_image import LoadImages
from utils.general import (
	check_img_size, non_max_suppression, apply_classifier, scale_coords,
	xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized


def get_resion_box(input_img, imgsz=416, weights="weights/kaimono/kaimono_0822/best_kaimono_e1000_20200822.pt", conf_thre =0.5):
	
	with torch.no_grad():
		# Initialize
		# set_logging()
		device = select_device("")
		half = device.type != 'cpu'  # half precision only supported on CUDA
		iou_thre  = 0.5

		# Load model
		model = attempt_load(weights, map_location=device)  # load FP32 model
		imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
		if half:
			model.half()  # to FP16

		# Second-stage classifier

		# Set Dataloader
		
		save_img = True
		dataset = LoadImages(input_img, img_size=imgsz)

		# Run inference
		img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
		_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
		rate_out = 0.0
		for path, img, im0s, vid_cap, rate in dataset:
			rate_out = rate
			img = torch.from_numpy(img).to(device)
			img = img.half() if half else img.float()  # uint8 to fp16/32
			img /= 255.0  # 0 - 255 to 0.0 - 1.0
			if img.ndimension() == 3:
				img = img.unsqueeze(0)

		# Inference
		pred = model(img, augment=False)[0]

		# Apply NMS
		pred = non_max_suppression(pred, conf_thre, iou_thre, classes=None, agnostic=False)

		# Apply Classifier
		region = []
		if (not pred[0] == None):
			region = pred[0].to('cpu').detach().numpy().copy()

		box_region = []
		for reg in region:

			if reg[-1] == 1:
				box_region = reg[0:4]//rate_out[0]

		print ("box_region",box_region)
		return box_region

def get_resion_item(input_img, imgsz=416, weights="weights/kaimono/kaimono_0822/best_kaimono_e1000_20200822.pt", conf_thre =0.5):
	
	with torch.no_grad():
		# Initialize
		# set_logging()
		device = select_device("")
		half = device.type != 'cpu'  # half precision only supported on CUDA
		iou_thre  = 0.5

		# Load model
		model = attempt_load(weights, map_location=device)  # load FP32 model
		imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
		if half:
			model.half()  # to FP16

		# Second-stage classifier

		# Set Dataloader
		dataset = LoadImages(input_img, img_size=imgsz)

		# Run inference
		img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
		_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
		rate_out = 0.0
		for path, img, im0s, vid_cap, rate in dataset:
			rate_out = rate
			img = torch.from_numpy(img).to(device)
			img = img.half() if half else img.float()  # uint8 to fp16/32
			img /= 255.0  # 0 - 255 to 0.0 - 1.0
			if img.ndimension() == 3:
				img = img.unsqueeze(0)

		# Inference
		pred = model(img, augment=False)[0]

		# Apply NMS
		pred = non_max_suppression(pred, conf_thre, iou_thre, classes=None, agnostic=False)

		# Apply Classifier
		region = []
		if (not pred[0] == None):
			region = pred[0].to('cpu').detach().numpy().copy()

		item_region = []
		for reg in region:

			if reg[-1] == 0:
				item_region.append(reg[0:4]//rate_out[0])

		print ("item_region",item_region)
		return item_region


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
	# Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
	shape = img.shape[:2]  # current shape [height, width]
	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)

	# Scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	if not scaleup:  # only scale down, do not scale up (for better test mAP)
		r = min(r, 1.0)

	# Compute padding
	ratio = r, r  # width, height ratios
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
	if auto:  # minimum rectangle
		dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
	elif scaleFill:  # stretch
		dw, dh = 0.0, 0.0
		new_unpad = (new_shape[1], new_shape[0])
		ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

	dw /= 2  # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad:  # resize
		img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
	return img, ratio, (dw, dh)
	   
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
	parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
	parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')
	parser.add_argument('--update', action='store_true', help='update all models')
	img = []
	conf_thre = 0.6
	# opt.iou_thre
	# opt.classes
	# opt.agnostic_nms
	input_img = cv2.imread("inference/images/img_0302.jpg")
	rate,region = get_resion_box(input_img,conf_thre=conf_thre)
	img2 = input_img[int(region[1]/rate) : int(region[3]/rate), int(region[0]/rate) : int(region[2]/rate)]

	cv2.imwrite("out_sample2.jpg", img2)
