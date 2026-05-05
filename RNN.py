# ==============================
# EXPERIMENT 11
# RNN vs LSTM vs GRU (IMDb Dataset)
# ==============================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Step 1: Import Libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 2: Load Dataset
print("Loading IMDb dataset...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Step 3: Preprocessing
max_len = 200
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ==============================
# Step 4: Vanilla RNN Model
# ==============================
print("\nTraining Vanilla RNN...")

rnn_model = Sequential([
    Embedding(input_dim=10000, output_dim=32, input_length=200),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

rnn_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

rnn_history = rnn_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

rnn_loss, rnn_acc = rnn_model.evaluate(X_test, y_test)
print("Vanilla RNN Accuracy:", rnn_acc)


# ==============================
# Step 5: LSTM Model
# ==============================
print("\nTraining LSTM...")

lstm_model = Sequential([
    Embedding(input_dim=10000, output_dim=32, input_length=200),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

lstm_history = lstm_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test)
print("LSTM Accuracy:", lstm_acc)


# ==============================
# Step 6: GRU Model
# ==============================
print("\nTraining GRU...")

gru_model = Sequential([
    Embedding(input_dim=10000, output_dim=32, input_length=200),
    GRU(32),
    Dense(1, activation='sigmoid')
])

gru_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

gru_history = gru_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

gru_loss, gru_acc = gru_model.evaluate(X_test, y_test)
print("GRU Accuracy:", gru_acc)


# ==============================
# Step 7: Comparison Graph
# ==============================
print("\nPlotting comparison graphs...")

plt.figure(figsize=(14, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(rnn_history.history['val_accuracy'], label='RNN')
plt.plot(lstm_history.history['val_accuracy'], label='LSTM')
plt.plot(gru_history.history['val_accuracy'], label='GRU')
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(rnn_history.history['val_loss'], label='RNN')
plt.plot(lstm_history.history['val_loss'], label='LSTM')
plt.plot(gru_history.history['val_loss'], label='GRU')
plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("rnn_comparison.png")
plt.show()


# ==============================
# Final Output Summary
# ==============================
print("\n===== FINAL RESULTS =====")
print(f"Vanilla RNN Accuracy: {rnn_acc:.4f}")
print(f"LSTM Accuracy      : {lstm_acc:.4f}")
print(f"GRU Accuracy       : {gru_acc:.4f}")