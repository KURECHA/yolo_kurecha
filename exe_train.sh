#!/bin/sh
python train.py --img 208 --batch 8 --epochs 1000 --data data/hacku_data.yaml --cfg models/yolov5x.yaml --name kaimono_e500_20200823 --weights weights/yolov5x.pt 