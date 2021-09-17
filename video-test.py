#!python

import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
	print("Cannot open camera")
	exit()
while True:
	# Capture frame by frame
	ret, frame = cap.read()
	
	# if frame correctly read
	if not ret:
		print("Cannot receive frame (stream end?) Exiting...")
		break
	# Our operations on the frame come here
	cv.imshow("Video", frame)
	if cv.waitKey(1) == ord("q"):
		break

# When eveyrthing done, release the capture
cap.release()
cv.destroyAllWindows()