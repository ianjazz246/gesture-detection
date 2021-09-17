#!/usr/bin/env python3

import numpy as np
import cv2 as cv
#from matplotlib import plyplot as plt

cap = cv.VideoCapture(0)
if not cap.isOpened():
	print("Cannot open camera")
	exit()
	
thresholdValue = 127
skinLowerBound = (87, 68, 71)
skinUpperBound = (191, 119, 128)

while True:
	# Capture frame by frame
	ret, frame = cap.read()
	
	# if frame correctly read
	if not ret:
		print("Cannot receive frame (stream end?) Exiting...")
		break

	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	blur = cv.blur(gray, (3, 3))
	ret, blur_thresh = cv.threshold(blur, 50, 255, cv.THRESH_BINARY)
	contours, hierachy = cv.findContours(blur_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	contourImg = frame.copy()

	for i in range(len(contours)):
		cv.drawContours(contourImg, contours, i, (255, 0, 0), thickness=3)

	ret,thresh1 = cv.threshold(frame, thresholdValue, 255, cv.THRESH_BINARY)
	thresh2 = cv.inRange(frame, skinLowerBound, skinUpperBound)

	# Our operations on the frame come here
	cv.imshow("Video", contourImg)
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