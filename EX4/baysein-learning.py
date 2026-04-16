# ==============================
# BAYESIAN LEARNING (NAIVE BAYES)
# ==============================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Generate Dataset
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=42
)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("\n--- Bayesian Learning (Naive Bayes) ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))

# Plot
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Naive Bayes Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()