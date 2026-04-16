# ==============================
# Chapter 7: Ensemble Learning
# ==============================

import sys
import sklearn
assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)

# ==============================
# 1. Dataset
# ==============================
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ==============================
# 2. Voting Classifier
# ==============================
log_clf = LogisticRegression()
rf_vote = RandomForestClassifier(n_estimators=50)
svm_clf = SVC(probability=True)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rf_vote), ('svc', svm_clf)],
    voting='soft'
)

voting_clf.fit(X_train, y_train)

print("\n--- Voting Accuracy ---")
print("Voting:", accuracy_score(y_test, voting_clf.predict(X_test)))

# ==============================
# 3. Models (FOR CLEAR GRAPH DIFFERENCE)
# ==============================

# VERY WEAK TREE
tree_clf = DecisionTreeClassifier(max_depth=1)
tree_clf.fit(X_train, y_train)

# BAGGING (moderate complexity)
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=50,
    max_samples=100,
    bootstrap=True
)
bag_clf.fit(X_train, y_train)

# RANDOM FOREST (stronger)
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None
)
rf_clf.fit(X_train, y_train)

print("\n--- Model Accuracy ---")
print("Decision Tree:", accuracy_score(y_test, tree_clf.predict(X_test)))
print("Bagging:", accuracy_score(y_test, bag_clf.predict(X_test)))
print("Random Forest:", accuracy_score(y_test, rf_clf.predict(X_test)))

# ==============================
# 4. Visualization Function
# ==============================

def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20)
    plt.title(title)

# ==============================
# 5. Decision Boundary Graph
# ==============================

plt.figure(figsize=(12, 4))

plt.subplot(131)
plot_decision_boundary(tree_clf, X, y, "Decision Tree (Very Weak)")

plt.subplot(132)
plot_decision_boundary(bag_clf, X, y, "Bagging")

plt.subplot(133)
plot_decision_boundary(rf_clf, X, y, "Random Forest")

plt.tight_layout()

# ==============================
# 6. Feature Importance
# ==============================

from sklearn.datasets import load_iris

iris = load_iris()
rf_iris = RandomForestClassifier(n_estimators=200)
rf_iris.fit(iris["data"], iris["target"])

plt.figure()
plt.barh(iris["feature_names"], rf_iris.feature_importances_)
plt.title("Feature Importance (Random Forest)")

# ==============================
# 7. Gradient Boosting Graph
# ==============================

from sklearn.ensemble import GradientBoostingRegressor

X_reg = np.random.rand(100, 1) - 0.5
y_reg = 3 * X_reg[:, 0]**2 + 0.05 * np.random.randn(100)

gbrt = GradientBoostingRegressor(n_estimators=100)
gbrt.fit(X_reg, y_reg)

x_plot = np.linspace(-0.5, 0.5, 100).reshape(-1, 1)
y_plot = gbrt.predict(x_plot)

plt.figure()
plt.scatter(X_reg, y_reg)
plt.plot(x_plot, y_plot)
plt.title("Gradient Boosting Regression")

# ✅ IMPORTANT: only ONE show()
plt.show()

# ==============================
print("\nAll models executed successfully!")