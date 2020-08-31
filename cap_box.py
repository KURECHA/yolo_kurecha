import get_region as gr
import cv2
import numpy as np

from bags_num import calc_bags_num


cap = cv2.VideoCapture(0)
#動画のプロパティの設定
cap.set(3, 800)
cap.set(4, 600)
sum_canny = 0
filter_size = 9
box_empty_edge_average = 10574595 
time = 0
box_size = 50

eco_bag_size = 31

weights_box="weights/kaimono/kaimono_item_weight/best_kaimono_e1000_20200822.pt"
weights_item="weights/kaimono/kaimono_item_weight/best_kaimono_e1000_20200822.pt"
conf_thre =0.5

while True:
	_, frame = cap.read()

	#YOLOを使ってかごの領域を検出→かごの領域を取得、
	region = gr.get_resion_box(frame, weights=weights_box, conf_thre=conf_thre)


	if not len(region) == 0 and min(region)>=0:
		# カゴの中心座標を取得
		frame_box = frame[int(region[1]) : int(region[3]), int(region[0]) : int(region[2])]
		frame_box_center = [n//2 for n in frame_box.shape][0:2]

		
		frame_gray = cv2.cvtColor(frame_box, cv2.COLOR_BGR2GRAY)
		dst = cv2.medianBlur(frame_gray, ksize=filter_size)

		frame_canny = cv2.Canny(dst, 50, 150)

		# カゴに入っている商品の座標を取得して黒で塗りつぶす
		region_item = gr.get_resion_item(frame_box, weights=weights_item, conf_thre=0.7)
		frame_canny_item_disable = []
		if len(region_item) != 0 :
			for reg in region_item:
				frame_canny_item_disable = cv2.rectangle(frame_canny, (int(reg[0]), int(reg[1]) ), ( int(reg[2]), int(reg[3])),  (0,0,0), -1)

		sum_canny = np.sum(frame_canny)

		k = cv2.waitKey(1000)
		time += 1

		
		if time >= 5:
			break
	else:
		time = 0

print(sum_canny)

# cv2.imshow('box', cv2.resize(frame, (600, 400)))
# k = cv2.waitKey(5000)
cv2.imshow('frame', cv2.resize(frame, (600, 400)))
k = cv2.waitKey(3000)
cv2.imshow('box', cv2.resize(frame_box, (600, 400)))
k = cv2.waitKey(3000)
cv2.imshow('box_edge', cv2.resize(frame_canny, (600, 400)))
k = cv2.waitKey(3000)
# cv2.imshow('', cv2.resize(frame_canny_item_disable, (600, 400)))
# k = cv2.waitKey(3000)
score = (1-sum_canny/box_empty_edge_average)*box_size

print(score)

eco_bag_capa = 40
bags_num = calc_bags_num(score, eco_bag_capa)
print("bags_num:\n(エコバッグ使用済み容量(%), S枚数, L枚数) = ", bags_num)

if score <= eco_bag_size:
	print("エコバッグのみで大丈夫です。")
else:
	print("レジ袋が必要です。")

#キャプチャを終了
cap.release()
cv2.destroyAllWindows()