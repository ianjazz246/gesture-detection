import cv2
import mediapipe as mp
from common import GESTURE_CATEGORIES, hand_landmark_to_model_input
import os
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

OUTPUT_ROOT_DIR = r"mediapipe"

# For images, but not used
currentFileNum = 1
selectedGestureI = 0
rows_added = 0

burst_capture_num = 0

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

def on_slider_change(new_value):
	global selectedGestureI, currentFileNum, rows_added
	global csv_file, writer
	selectedGestureI = new_value
	csv_file.close()
	csv_file = open(
		os.path.join(OUTPUT_ROOT_DIR, GESTURE_CATEGORIES[selectedGestureI], GESTURE_CATEGORIES[selectedGestureI] + ".csv"),
		mode="a+", newline=""
	)
	writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

	currentFileNum = get_next_filenames() + 1
	rows_added = get_csv_rows()


create_folders()
currentFileNum = get_next_filenames() + 1
rows_added = get_csv_rows()

csv_file = open(
	os.path.join(OUTPUT_ROOT_DIR, GESTURE_CATEGORIES[selectedGestureI], GESTURE_CATEGORIES[selectedGestureI] + ".csv"),
	mode="a+", newline=""
)
writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("MediaPipe Hands")
cv2.createTrackbar("Category", "MediaPipe Hands", selectedGestureI, len(GESTURE_CATEGORIES) - 1, on_slider_change)


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
			cv2.rectangle(image, (20, 5), (150, 35), (255, 255, 255), thickness = -1)
			cv2.putText(image, str(results.multi_hand_landmarks[0].landmark[0]), (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))

		cv2.putText(image, GESTURE_CATEGORIES[selectedGestureI], (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
		cv2.imshow('MediaPipe Hands', image)
		key = cv2.waitKey(5)
		# Escape key
		if key & 0xFF == 27:
			break
		if key == ord('b'):
			burst_capture_num = 10
		if key == ord('s') or burst_capture_num > 0:
			if results.multi_hand_landmarks:
				#cv2.imwrite(os.path.join(OUTPUT_ROOT_DIR, GESTURE_CATEGORIES[selectedGestureI], str(currentFileNum) + ".png"), original_image)

				output_row = []
				first_hand_landmarks = results.multi_hand_landmarks[0].landmark
				for point in first_hand_landmarks:
					output_row.append([point.x, point.y, point.z])

				writer.writerow(output_row)

				rows_added += 1
				burst_capture_num -= 1
				print("Added {} rows".format(rows_added))
			pass

csv_file.close()
cap.release()
cv2.destroyAllWindows()