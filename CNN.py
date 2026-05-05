import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Paths
DATASET_PATH = r"C:\Users\Sanika\OneDrive\Desktop\ML-Lab\EX7\dataset\train"
LABEL_PATH = r"C:\Users\Sanika\OneDrive\Desktop\ML-Lab\EX7\dataset\trainLabels.csv"

print("Loading dataset...")

# Load labels
labels_df = pd.read_csv(LABEL_PATH)

images = []
labels = []

# Read images
for i, row in labels_df.iterrows():
    img_name = str(row['id']) + ".png"
    img_path = os.path.join(DATASET_PATH, img_name)

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32, 32))
        images.append(img)
        labels.append(row['label'])

# Convert to numpy
X = np.array(images) / 255.0
y = np.array(labels)

print("Total images loaded:", len(X))

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nTraining model...")
history = model.fit(X_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_split=0.2)

print("\nEvaluating model...")
loss, acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", acc)