import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import xyxy2xywh, xywh2xyxy, torch_distributed_zero_first

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

class LoadImages:  # for inference
	def __init__(self,img,  img_size=640):

		self.input_img = img
		self.img_size = img_size
		self.nf = 2  # number of files
		i = 0
		self.cap = 0

	def __iter__(self):
		self.count = 0
		return self

	def __next__(self):
		if self.count == self.nf:
			raise StopIteration
		# Read image
		self.count += 1
		img0 = self.input_img  # BGR

		# Padded resize
		img,rate, _ = letterbox(img0, new_shape=self.img_size)

		# Convert
		img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
		img = np.ascontiguousarray(img)

		path=0
		return path, img, img0, self.cap, rate

	def new_video(self, path):
		self.frame = 0
		self.cap = cv2.VideoCapture(path)
		self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

	def __len__(self):
		return self.nf  # number of files

