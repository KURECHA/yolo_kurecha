#!/bin/sh
python detect.py --weights weights/kaimono/kaimono_box_weight/best_kaimono_e1000_20200822.pt  --img 416 --conf 0.5 --source 0  --device 'cpu'