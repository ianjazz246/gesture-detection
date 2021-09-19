import itertools
import numpy as np
from scipy.spatial import distance

GESTURE_CATEGORIES = ["other", "fist", "flatpalm", "one", "two", "three", "four", "ok"]

# Pass in results from mediapipe hand
def hand_landmark_to_model_input(results):
	points = []
	first_hand_landmarks = results.multi_hand_landmarks[0].landmark
	for point in first_hand_landmarks:
		points.append(np.array([point.x, point.y, point.z]))

	wrist_to_index_base_dist = distance.euclidean(points[0], points[5])

	return [np.linalg.norm(b - a) / wrist_to_index_base_dist for a, b in itertools.combinations(points, 2)]