# ==========================================
# EXPERIMENT 08
# CNN for Video Classification (FINAL FIXED)
# ==========================================

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Input

# ==========================================
# STEP 1: PARAMETERS
# ==========================================

DATASET_PATH = r"C:\Users\Sanika\OneDrive\Desktop\ML-Lab\EX8\dataset"
IMG_SIZE = 64
MAX_FRAMES = 10   # frames per video

# ==========================================
# STEP 2: LOAD DATA (VIDEO → FRAMES)
# ==========================================

data = []
labels = []
class_names = []

print("Loading videos and extracting frames...")

# Check dataset exists
if not os.path.exists(DATASET_PATH):
    print("❌ Dataset path not found!")
    exit()

for label, folder in enumerate(os.listdir(DATASET_PATH)):

    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    class_names.append(folder)

    for video_file in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_file)

        cap = cv2.VideoCapture(video_path)
        count = 0

        while cap.isOpened() and count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0

            data.append(frame)
            labels.append(label)

            count += 1

        cap.release()

print("Total frames:", len(data))

# Convert to numpy
X = np.array(data)
y = np.array(labels)

# ==========================================
# STEP 3: TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ==========================================
# STEP 4: BUILD CNN MODEL (FIXED)
# ==========================================

print("\nBuilding CNN model...")

model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),   # ✅ FIXED (no warning)

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# STEP 5: TRAIN MODEL
# ==========================================

print("\nTraining model...")

history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# ==========================================
# STEP 6: EVALUATION
# ==========================================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(X_test, y_test)

print("\n✅ Test Accuracy:", accuracy)

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ==========================================
# STEP 7: PLOTS
# ==========================================

# Accuracy plot
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss plot
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()