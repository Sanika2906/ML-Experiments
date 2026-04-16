# ==============================
# Step 1: Import Libraries
# ==============================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import make_classification

# ==============================
# Step 2: Generate Sample Data
# ==============================

np.random.seed(42)

# Regressor Data (Sine Wave)
X_reg = 10 * np.random.rand(100, 1)
y_reg = np.sin(X_reg).ravel() + np.random.randn(100) * 0.1

# Classifier Data (FIXED LINE)
X_clf, y_clf = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=42
)

# ==============================
# Step 3: Train-Test Split
# ==============================

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# ==============================
# Step 4: Model Training
# ==============================

# Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = rf_regressor.predict(X_test_reg)

# Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_clf, y_train_clf)
y_pred_clf = rf_classifier.predict(X_test_clf)

# ==============================
# Step 5: Evaluation
# ==============================

print("\n--- Model Evaluation ---")

mse = mean_squared_error(y_test_reg, y_pred_reg)
print("Random Forest Regressor MSE:", round(mse, 2))

accuracy = accuracy_score(y_test_clf, y_pred_clf)
print("Random Forest Classifier Accuracy:", round(accuracy, 2))

# ==============================
# Step 6: Visualization
# ==============================

# 🔹 Regression Plot
x_plot = np.linspace(0, 10, 100).reshape(-1, 1)
y_plot = rf_regressor.predict(x_plot)

plt.figure()
plt.scatter(X_reg, y_reg)
plt.plot(x_plot, y_plot)
plt.title("Random Forest Regression")
plt.xlabel("X")
plt.ylabel("y")

# 🔹 Classification Plot
plt.figure()
plt.scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf)
plt.title("Random Forest Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Show all graphs together
plt.show()

# ==============================
print("\nAll models executed successfully!")