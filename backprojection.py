#!/usr/bin/env python3

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random

is_calibrating = False

CALIBRATE_RECT_WIDTH = 10
CALIBRATE_RECT_HEIGHT = 25
CALIBRATE_RECT_ROWS = 3
CALIBRATE_RECT_COLS = 3
# Used in create_calibrate_rects. Rects spaced by 1/CALIBRATE_RECT_DIVIDES 
CALIBRATE_RECT_WIDTH_DIVIDES = 21
CALIBRATE_RECT_HEIGHT_DIVIDES = 7

HIST_MASKING_MIN_VALUE = 130

CANNY_MIN_THRESH = 50
CANNY_MAX_THRESH = 100

PRIMARY_WINDOW_NAME = "Video"
	
thresholdValue = 127
recordedSkinColor = (100, 100, 100)
skinLowerBound = (87, 68, 71)
skinUpperBound = (191, 119, 128)

roi_rects = []

slider_value = CALIBRATE_RECT_HEIGHT_DIVIDES

def create_calibrate_rects(img):
	rect_count = CALIBRATE_RECT_ROWS * CALIBRATE_RECT_COLS
	width = len(img[0])
	height = len(img)

	new_roi_rects = []

	# top left, bottom right
	for row in range(CALIBRATE_RECT_ROWS):
		for col in range(CALIBRATE_RECT_COLS):
			# [y, x] for two opposite vertices
			rect = np.array([
				[
					width / CALIBRATE_RECT_WIDTH_DIVIDES * (row + CALIBRATE_RECT_WIDTH_DIVIDES // 2 - 1) - CALIBRATE_RECT_WIDTH / 2,
					height / CALIBRATE_RECT_HEIGHT_DIVIDES * (col + CALIBRATE_RECT_HEIGHT_DIVIDES // 2 - 1) - CALIBRATE_RECT_HEIGHT / 2,
				],
				[
					width / CALIBRATE_RECT_WIDTH_DIVIDES * (row + CALIBRATE_RECT_WIDTH_DIVIDES // 2 - 1) + CALIBRATE_RECT_WIDTH / 2,
					height / CALIBRATE_RECT_HEIGHT_DIVIDES * (col + CALIBRATE_RECT_HEIGHT_DIVIDES // 2 - 1) + CALIBRATE_RECT_HEIGHT / 2,
				]
			], dtype=np.uint32)

			new_roi_rects.append(rect)
	
	global roi_rects
	roi_rects = new_roi_rects

def draw_calibrate_rects(img):
	for rect in roi_rects:
		cv.rectangle(img, rect[0], rect[1], (255, 0, 0))

# Returns histogram
def get_histrogram_from_roi(img, rects):
	hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

	print(rects)
	rect_height = rects[0][1][1] - rects[0][0][1]
	rect_width = rects[0][1][0] - rects[0][0][0]

	roi = np.zeros([len(rects) * rect_height, rect_width, 3], dtype=hsv.dtype)

	for i in range(len(rects)):
		roi[i * rect_height : i * rect_height + rect_height, :] = hsv[rects[i][0][1] : rects[i][1][1], rects[i][0][0] : rects[i][1][0]]

	cv.imshow("ROI", cv.cvtColor(roi, cv.COLOR_HSV2BGR))

	# Quantize hue (0th channel) into 180 levels, with range from 0 to 180
	# Quantize saturation (1st channel) into 256 levels, with range from 0 to 256
	hist = cv.calcHist([roi], [0, 1], None, [90, 128], [0, 180, 0, 256])
	cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)

	return hist

# Image should be in same color format as histogram was calculated with
# Returns mask and masked image
def hist_masking(img, hist):
	dst = cv.calcBackProject([img], [0, 1], hist, [0, 180, 0, 256], 1)

	# Convoluite with circular disc
	disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
	cv.filter2D(dst, -1, disc, dst)

	ret, thresh = cv.threshold(dst, HIST_MASKING_MIN_VALUE, 255, cv.THRESH_BINARY)

	thresh_3channel = cv.merge((thresh, thresh, thresh))

	return (thresh, cv.bitwise_and(img, thresh_3channel))

def get_hand_features(img, draw_img):
	contours, hierachy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	for i in range(len(contours)):
		cv.drawContours(draw_img, contours, i, (255, 0, 0))

def edge_contours(img):
	draw_img = img.copy()
	edges = cv.Canny(img, CANNY_MIN_THRESH, CANNY_MAX_THRESH)
	contours, hierachy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	for i in range(len(contours)):
		color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
		cv.drawContours(draw_img, contours, i, color)
	cv.imshow("Edge contour", draw_img)

def on_slider_change(new_value):
	slider_value = new_value
	global CALIBRATE_RECT_HEIGHT_DIVIDES
	CALIBRATE_RECT_HEIGHT_DIVIDES = new_value
	create_calibrate_rects(frame)


	
print("Opening camera...")
try:
	cap = cv.VideoCapture(0, cv.CAP_DSHOW)
except:
	cap = cv.VideoCapture(0)
print("Camera opened")
if not cap.isOpened():
	print("Cannot open camera")
	exit()

for i in range(2):
	ret, frame = cap.read()
	if ret:
		break
	print("Error capturing frames")

print("Successfuly retireved frame")

print(len(frame))
print(len(frame[0]))

create_calibrate_rects(frame)

cv.namedWindow(PRIMARY_WINDOW_NAME)
cv.createTrackbar("slider", PRIMARY_WINDOW_NAME, slider_value, 100, on_slider_change)

while True:
	# Capture frame by frame
	ret, frame = cap.read()
	# cv.imshow("Video", frame)
	
	# if frame correctly read
	if not ret:
		print("Cannot receive frame (stream end?) Exiting...")
		break

	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

	frame_cpy = frame.copy()
	if is_calibrating:	
		draw_calibrate_rects(frame_cpy)
	
	cv.imshow(PRIMARY_WINDOW_NAME, frame_cpy)

	key = cv.waitKey(1)
	if key == ord("q"):
		break
	if key == ord("+"):
		thresholdValue += 1
	if key == ord("-"):
		thresholdValue -= 1
	if key == ord("c"):
		is_calibrating = not is_calibrating
	if key == ord("s"):
		cv.imwrite("capture.png", frame)
	if key == ord("v"):
		cv.imshow("Original", frame)
		hist = get_histrogram_from_roi(frame, roi_rects)
		thresh, masked = hist_masking(hsv, hist)
		cv.imshow("Masked", masked)
		get_hand_features(thresh, frame_cpy)
		cv.imshow("Contours", frame_cpy)
		edge_contours(frame)
	if key == ord("b"):
		cv.imshow("Original", frame)
		thresh, masked = hist_masking(hsv, hist)
		cv.imshow("Masked", masked)
		get_hand_features(thresh, frame_cpy)
		cv.imshow("Contours", frame_cpy)
		edge_contours(frame)

		

# When eveyrthing done, release the capture
cap.release()
cv.destroyAllWindows()