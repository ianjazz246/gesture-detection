import csv
import os
import keras
import numpy as np
import itertools
import math
from common import GESTURE_CATEGORIES, hand_landmark_to_model_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

MEDIAPIPE_DIR = "mediapipe"
NUM_LANDMARKS = 21
BATCH_SIZE = 32
EPOCHS = 10

hand_coordinates = []
y = []

for i, gesture in enumerate(GESTURE_CATEGORIES):
	with open(os.path.join(MEDIAPIPE_DIR, gesture, gesture + ".csv"), newline="") as csv_file:
		reader = csv.reader(csv_file, delimiter=',', quotechar='"')
		addedRows = 0
		for row in reader:
			if addedRows >= 500:
				break
			row_out = []
			for j in range(0, len(row), 3):
				row_out.append([float(x) for x in row[j : j + 3]])
			hand_coordinates.append(row_out)
			y.append(i)
			addedRows += 1

hand_coordinates = np.array(hand_coordinates)
X = []

# Get distance between every landmark
for hand in hand_coordinates:
	# linalg.norm is euclidiean distance
	X.append([np.linalg.norm(b - a) for a, b in itertools.combinations(hand, 2)])

X = np.array(X)
y = np.array(y)

y = to_categorical(y)

print(X.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

model = Sequential()

model.add(Dense(100, activation="relu", input_shape=(math.comb(NUM_LANDMARKS, 2),)))
model.add(Dense(25, activation="relu"))
model.add(Dense(4, activation="softmax"))

print(model.summary())

optimizer = optimizers.Adam(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='mediapipe/best.h5',
				save_best_only = True,
				monitor="val_loss",
		)
]

history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test), callbacks=callbacks)

y_pred = model.predict(X_test, batch_size=BATCH_SIZE)

matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=GESTURE_CATEGORIES)
disp.plot()
plt.show()