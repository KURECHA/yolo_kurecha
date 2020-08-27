
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


def get_region(input_img, imgsz=416, weights="weights/kaimono/kaimono_0822/best_kaimono_e1000_20200822.pt"):
	
	# Initialize
	set_logging()
	device = select_device("")
	half = device.type != 'cpu'  # half precision only supported on CUDA

	# Load model
	model = attempt_load(weights, map_location=device)  # load FP32 model
	imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
	if half:
		model.half()  # to FP16

	# Second-stage classifier

	# Set Dataloader
	
	save_img = True
	dataset = LoadImages(input_img, img_size=imgsz)
	print("######################")
	print(dataset)
	print("####################")

	# Run inference
	img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
	_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
	rate_out = 0.0
	for path, img, im0s, vid_cap, rate in dataset:
		rate_out = rate
		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		print("##################shappe##########")
		print(img.shape)
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

	# Inference
	print("#####################imgshape", img.shape)
	pred = model(img, augment=False)[0]

	# Apply NMS
	pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
	print("###############pred")
	print(pred)

	# Apply Classifier
	print("#########################")
	region = []
	if pred[0] != None:	
		region = pred[0].to('cpu').detach().numpy().copy()
	print("reg",region)

	box_region = []
	for reg in region:

		if reg[-1] == 1:
			box_region = reg[0:4]
	print("#########################")
	print (box_region)
	return rate_out,box_region


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
	opt = parser.parse_args()
	print(opt)
	img = []
	cap = cv2.VideoCapture(0)
	#動画のプロパティの設定
	cap.set(3, 800)
	cap.set(4, 600)
	sum_canny = 0
	filter_size = 9
	box_empty_edge_average = 6110565
	time = 0
	box_size = 15

	eco_bag_size = 10

	while True:
		_, frame = cap.read()

		with torch.no_grad():
			# input_img = cv2.imread("inference/images/IMG_0302.JPG")
			# input_img = letterbox(input_img, new_shape=416)[0]
			# input_img = input_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
			# rate,region = detect(input_img)
			rate, region = get_region(frame)
			


			if not len(region) == 0:
				rate = rate[0]
				frame_box = frame[int(region[1]/rate) : int(region[3]/rate), int(region[0]/rate) : int(region[2]/rate)]

				# cv2.imwrite("out_sample2.jpg", img2)

				frame_gray = cv2.cvtColor(frame_box, cv2.COLOR_BGR2GRAY)
				# :w
				# dst = cv2.medianBlur(frame_gray, ksize=filter_size)

				frame_canny = cv2.Canny(frame_gray, 50, 150)
				sum_canny = np.sum(frame_canny)

				k = cv2.waitKey(1000)
				time += 1

				
				if time >= 10:
					break

			else:
				time = 0

	print(sum_canny)

	
	cv2.imshow('PUSH ENTER KEY', cv2.resize(frame, (600, 400)))
	k = cv2.waitKey(5000)
	score = (1-sum_canny/box_empty_edge_average)*box_size

	print(box_size)

	if score <= eco_bag_size:
		print("エコバッグのみで大丈夫です。")
	else:
		print("レジ袋が必要です。")

	#キャプチャを終了
	cap.release()
	cv2.destroyAllWindows()
