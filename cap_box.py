from get_region import get_resion_box
import cv2
import numpy as np



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

    #YOLOを使ってかごの領域を検出→かごの領域を取得、
    region = get_resion_box(frame)


    if not len(region) == 0:
        frame_box = frame[int(region[1]) : int(region[3]), int(region[0]) : int(region[2])]


        frame_gray = cv2.cvtColor(frame_box, cv2.COLOR_BGR2GRAY)
        # dst = cv2.medianBlur(frame_gray, ksize=filter_size)

        frame_canny = cv2.Canny(frame_gray, 50, 150)
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
cv2.imshow('box', cv2.resize(frame_box, (600, 400)))
k = cv2.waitKey(5000)
cv2.imshow('box_edge', cv2.resize(frame_canny, (600, 400)))
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