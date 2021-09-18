#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import random
#from matplotlib.animation import FuncAnimation
#from matplotlib import pyplot as plt

cap = cv.VideoCapture(0)
if not cap.isOpened():
	print("Cannot open camera")
	exit()
	
thresholdValue = 127
skin_ycrcb_mint = np.array((0, 133, 77))
skin_ycrcb_maxt = np.array((255, 173, 127))


while True:
	ret, frame = cap.read()

	# if frame correctly read
	if not ret:
		print("Cannot receive frame (stream end?) Exiting...")
		break

	skin_ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)

	blur = cv.blur(skin_ycrcb, (3, 3))
	_, thresh = cv.threshold(skin_ycrcb, 30, 255, cv.THRESH_BINARY)

	cv.imshow("Threshold", thresh)

	edges = cv.Canny(frame, 100, 200)
	contours, hierachy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	hull_list = []
	for i in range(len(contours)):
		hull = cv.convexHull(contours[i])
		hull_list.append(hull)

	contourImg = frame.copy()
	for i in range(len(contours)):
		color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
		cv.drawContours(contourImg, contours, i, color, thickness=3)
		cv.drawContours(contourImg, hull_list, i, color, thickness=1)

	# Our operations on the frame come here
	cv.imshow("Video", frame)
	cv.imshow("Contours", contourImg)

	key = cv.waitKey(1)
	if key == ord("q"):
		break
	if key == ord("+"):
		thresholdValue += 1
	if key == ord("-"):
		thresholdValue -= 1

# When eveyrthing done, release the capture
cap.release()
cv.destroyAllWindows()