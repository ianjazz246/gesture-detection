import cv2
import mediapipe as mp
from common import GESTURE_CATEGORIES, hand_landmark_to_model_input
import os
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

OUTPUT_ROOT_DIR = r"C:\Users\iansw\source\repos\gesture-detection\mediapipe"

# For images, but not used
currentFileNum = 1
selectedGestureI = 3
rows_added = 0

def create_folders():
	for category in GESTURE_CATEGORIES:
		folder_path = os.path.join(OUTPUT_ROOT_DIR, category)
		if not os.path.isdir(folder_path):
			os.mkdir(folder_path)
		try:
			with open(os.path.join(folder_path, category + ".csv"), mode="x"):
				pass
		except:
			pass

def get_next_filenames():
	path = os.path.join(OUTPUT_ROOT_DIR, GESTURE_CATEGORIES[selectedGestureI])
	maxFileNum = 0
	with os.scandir(path) as dirs:
		for entry in dirs:
			if entry.is_file():
				try:
					# Get filename without extension
					fileNum = int(os.path.splitext(entry.name)[0])
					if fileNum > maxFileNum:
						maxFileNum = fileNum
				except:
					pass
	return maxFileNum

def get_csv_rows():
	num_lines = sum(1 for line in open(os.path.join(OUTPUT_ROOT_DIR, GESTURE_CATEGORIES[selectedGestureI], GESTURE_CATEGORIES[selectedGestureI] + ".csv")))
	return num_lines

create_folders()
currentFileNum = get_next_filenames() + 1
rows_added = get_csv_rows()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
with open(
					os.path.join(OUTPUT_ROOT_DIR, GESTURE_CATEGORIES[selectedGestureI], GESTURE_CATEGORIES[selectedGestureI] + ".csv"),
					mode="a+", newline="") as csv_file:
	writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	with mp_hands.Hands(
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5) as hands:
		while cap.isOpened():
			success, original_image = cap.read()
			if not success:
				print("Ignoring empty camera frame.")
				# If loading a video, use 'break' instead of 'continue'.
				continue

			# Flip the image horizontally for a later selfie-view display, and convert
			# the BGR image to RGB.
			image = cv2.cvtColor(cv2.flip(original_image, 1), cv2.COLOR_BGR2RGB)
			# To improve performance, optionally mark the image as not writeable to
			# pass by reference.
			image.flags.writeable = False
			results = hands.process(image)

			# Draw the hand annotations on the image.
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			if results.multi_hand_landmarks:
				for hand_landmarks in results.multi_hand_landmarks:
					mp_drawing.draw_landmarks(
							image,
							hand_landmarks,
							mp_hands.HAND_CONNECTIONS,
							mp_drawing_styles.get_default_hand_landmarks_style(),
							mp_drawing_styles.get_default_hand_connections_style())
				cv2.putText(image, str(results.multi_hand_landmarks[0].landmark[0]), (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))

			cv2.putText(image, GESTURE_CATEGORIES[selectedGestureI], (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
			cv2.imshow('MediaPipe Hands', image)
			key = cv2.waitKey(5)
			# Escape key
			if key & 0xFF == 27:
				break
			if key == ord('s'):
				if results.multi_hand_landmarks[0]:
					#cv2.imwrite(os.path.join(OUTPUT_ROOT_DIR, GESTURE_CATEGORIES[selectedGestureI], str(currentFileNum) + ".png"), original_image)

					output_row = []
					first_hand_landmarks = results.multi_hand_landmarks[0].landmark
					for point in first_hand_landmarks:
						output_row.append([point.x, point.y, point.z])

					writer.writerow(output_row)

					rows_added += 1
					print("Added {} rows".format(rows_added))
				pass
cap.release()
cv2.destroyAllWindows()