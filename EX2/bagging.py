# ==============================
# BAGGING vs BOOSTING (COMBINED)
# ==============================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ==============================
# 1. Load Data (REDUCED SIZE)
# ==============================

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)

# 🔥 Reduce dataset for smooth execution
X = X[:10000]
y = y[:10000]

# Train-Test Split
X_train, X_test = X[:8000], X[8000:]
y_train, y_test = y[:8000], y[8000:]

# ==============================
# 2. BAGGING MODEL
# ==============================

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=5,
    max_samples=0.8,
    bootstrap=True,
    random_state=42
)

bag_clf.fit(X_train, y_train)
y_pred_bag = bag_clf.predict(X_test)

bag_acc = accuracy_score(y_test, y_pred_bag)

# ==============================
# 3. BOOSTING MODEL
# ==============================

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=20,
    learning_rate=0.5,
    random_state=42
)

ada_clf.fit(X_train, y_train)
y_pred_boost = ada_clf.predict(X_test)

boost_acc = accuracy_score(y_test, y_pred_boost)

# ==============================
# 4. PRINT RESULTS
# ==============================

print("\n--- Model Comparison ---")
print("Bagging Accuracy :", round(bag_acc, 3))
print("Boosting Accuracy:", round(boost_acc, 3))

# ==============================
# 5. CONFUSION MATRICES
# ==============================

cm_bag = confusion_matrix(y_test, y_pred_bag)
cm_boost = confusion_matrix(y_test, y_pred_boost)

cm_bag_norm = cm_bag / cm_bag.sum(axis=1, keepdims=True)
cm_boost_norm = cm_boost / cm_boost.sum(axis=1, keepdims=True)

# ==============================
# 6. GRAPH (SIDE-BY-SIDE)
# ==============================

plt.figure(figsize=(12,5))

# Bagging Plot
plt.subplot(1, 2, 1)
sns.heatmap(cm_bag_norm, cmap="YlGnBu")
plt.title("Bagging Confusion Matrix")

# Boosting Plot
plt.subplot(1, 2, 2)
sns.heatmap(cm_boost_norm, cmap="coolwarm")
plt.title("Boosting Confusion Matrix")

plt.tight_layout()

# SAVE IMAGE (IMPORTANT)
plt.savefig("bagging_vs_boosting.png")

plt.show(block=True)

# ==============================
print("\nGraph saved as: bagging_vs_boosting.png")
print("All models executed successfully!")