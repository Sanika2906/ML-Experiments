# ==============================
# BOOSTING (MNIST - AdaBoost)
# ==============================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Data
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)

# Train-Test Split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# AdaBoost Model
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),  # weak learner
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

# Train
ada_clf.fit(X_train, y_train)

# Predict
y_pred = ada_clf.predict(X_test)

# Accuracy
print("\n--- Boosting Accuracy (AdaBoost) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(8,6))
sns.heatmap(cm_norm, cmap="coolwarm")
plt.title("Boosting Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()