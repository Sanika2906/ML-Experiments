# ==============================
# EXPERIMENT 10
# Sentiment Analysis using RNN
# ==============================

# Step 1: Import Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hide warnings

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 2: Load Dataset
print("Loading IMDb dataset...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# Step 3: Preprocess Data
max_len = 200

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

print("After padding:")
print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# Step 4: Build RNN Model
print("\nBuilding RNN model...")

model = Sequential([
    Embedding(input_dim=10000, output_dim=32),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

# Step 5: Compile Model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Step 6: Train Model
print("\nTraining model...")

history = model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

# Step 7: Evaluate Model
print("\nEvaluating model...")

loss, accuracy = model.evaluate(X_test, y_test)

print("\n✅ Test Accuracy:", accuracy)
print("✅ Test Loss:", loss)

# Step 8: Plot Accuracy Graph
plt.figure(figsize=(10, 4))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Step 9: Plot Loss Graph
plt.figure(figsize=(10, 4))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Step 10: Sample Prediction
print("\nTesting on sample data...")

sample_index = 0
sample = X_test[sample_index].reshape(1, -1)

prediction = model.predict(sample)

if prediction > 0.5:
    print("Predicted Sentiment: Positive")
else:
    print("Predicted Sentiment: Negative")