import itertools
import numpy as np

GESTURE_CATEGORIES = ["fist", "flatpalm", "one", "two", "three", "four", "ok"]

# Pass in results from mediapipe hand
def hand_landmark_to_model_input(results):
	points = []
	first_hand_landmarks = results.multi_hand_landmarks[0].landmark
	for point in first_hand_landmarks:
		points.append(np.array([point.x, point.y, point.z]))

	return [np.linalg.norm(b - a) for a, b in itertools.combinations(points, 2)]