#!/usr/bin/env python3

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

is_calibrating = False

cap = cv.VideoCapture(0)
if not cap.isOpened():
	print("Cannot open camera")
	exit()

CALIBRATE_RECT_WIDTH = 20
CALIBRATE_RECT_HEIGHT = 20
CALIBRATE_RECT_ROWS = 3
CALIBRATE_RECT_COLS = 3
	
thresholdValue = 127
recordedSkinColor = (100, 100, 100)
skinLowerBound = (87, 68, 71)
skinUpperBound = (191, 119, 128)

roi_rects = []

def calibrate_skin(img, x, y, left, right):
	img_crop = img[x:x+left, y:y+right]

def create_calibrate_rects(img):
	rect_count = CALIBRATE_RECT_ROWS * CALIBRATE_RECT_COLS
	width = len(img)
	height = len(img[0])

	new_roi_rects = []

	# top left, bottom right
	for row in range(CALIBRATE_RECT_ROWS):
		for col in range(CALIBRATE_RECT_COLS):
			rect = np.array([
				[height / 9 * (col + 4) - CALIBRATE_RECT_HEIGHT / 2, width / 9 * (row + 4) - CALIBRATE_RECT_WIDTH / 2],
				[height / 9 * (col + 4) + CALIBRATE_RECT_HEIGHT / 2, width / 9 * (row + 4)  + CALIBRATE_RECT_WIDTH / 2]
			], dtype=np.uint32)

			new_roi_rects.append(rect)
	
	global roi_rects
	roi_rects = new_roi_rects

def draw_calibrate_rects(img):
	for rect in roi_rects:
		cv.rectangle(img, rect[0], rect[1], (255, 0, 0))

def get_histrogram_from_roi(img, rects):
	img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

	print(rects)
	rect_height = rects[0][1][0] - rects[0][0][0]
	rect_width = rects[0][1][1] - rects[0][0][1]

	roi = np.zeros([len(rects) * rect_height, rect_width, 3], dtype=img.dtype)

	for i in len(rects):
		roi[i * rect_height : i * rect_height + rect_height, :] = img[rects[i][0][1] : rects[i][1][1], rects[i][0][0] : rects[i][1][0]]

	cv2.imshow("ROI", roi)

	
ret, frame = cap.read()
print(len(frame))
print(len(frame[0]))
create_calibrate_rects(frame)

while True:
	# Capture frame by frame
	ret, frame = cap.read()
	# cv.imshow("Video", frame)
	
	# if frame correctly read
	if not ret:
		print("Cannot receive frame (stream end?) Exiting...")
		break

	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

	if is_calibrating:
		draw_calibrate_rects(frame)
	
	cv.imshow("Video with calibration", frame)


	key = cv.waitKey(1)
	if key == ord("q"):
		break
	if key == ord("+"):
		thresholdValue += 1
	if key == ord("-"):
		thresholdValue -= 1
	if key == ord("c"):
		is_calibrating = not is_calibrating
	if key == ord("v"):
		get_histrogram_from_roi(frame, roi_rects)

# When eveyrthing done, release the capture
cap.release()
cv.destroyAllWindows()