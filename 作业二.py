#############################################Decision Trees###################################################
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load data
train_df = pd.read_csv("training_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train = train_df[["X", "Y", "Z"]].values
y_train = train_df["Label"].values
X_test = test_df[["X", "Y", "Z"]].values
y_test = test_df["Label"].values

# 2. Train Decision Tree
clf = DecisionTreeClassifier(max_depth=10,random_state=42)
clf.fit(X_train, y_train)

# 3. Predictions
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# 4. Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}\n")

# 5. Classification Report
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred))


# 7. 3D Plot of Test Set Predictions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test_pred, cmap='viridis', marker='o')
legend1 = ax.legend(*scatter.legend_elements(), title="Predicted")
ax.add_artist(legend1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("3D Decision Tree Predictions (Test Set)")
plt.show()

# ✅ 8. 3D Plot of Training Set Predictions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train_pred, cmap='viridis', marker='o')
legend1 = ax.legend(*scatter.legend_elements(), title="Predicted")
ax.add_artist(legend1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("3D Decision Tree Predictions (Training Set)")
plt.show()











####################################################################adaboost+decisonTree#################################################################
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load data
train_df = pd.read_csv("training_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train = train_df[["X", "Y", "Z"]].values
y_train = train_df["Label"].values
X_test = test_df[["X", "Y", "Z"]].values
y_test = test_df["Label"].values

# 2. Create AdaBoost model with DecisionTree base estimator
base_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
ada_clf = AdaBoostClassifier(estimator=base_tree, n_estimators=1000, random_state=42)
ada_clf.fit(X_train, y_train)

# 3. Predictions
y_train_pred = ada_clf.predict(X_train)
y_test_pred = ada_clf.predict(X_test)

# 4. Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}\n")

# 5. Classification Report
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred))


# 7. 3D Plot of Test Set Predictions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test_pred, cmap='viridis', marker='o')
legend1 = ax.legend(*scatter.legend_elements(), title="Predicted")
ax.add_artist(legend1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("3D AdaBoost Predictions (Test Set)")
plt.show()

# 8. 3D Plot of Training Set Predictions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train_pred, cmap='viridis', marker='o')
legend1 = ax.legend(*scatter.legend_elements(), title="Predicted")
ax.add_artist(legend1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("3D AdaBoost Predictions (Training Set)")
plt.show()






















##############################################################################       SVM          ######################################################
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load data
train_df = pd.read_csv("training_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train = train_df[["X", "Y", "Z"]].values
y_train = train_df["Label"].values
X_test = test_df[["X", "Y", "Z"]].values
y_test = test_df["Label"].values

# 2. Train SVM
svm_clf = SVC(kernel="sigmoid", gamma="scale", C=1.0,coef0=0.1)
svm_clf.fit(X_train, y_train)

# 3. Predictions
y_train_pred = svm_clf.predict(X_train)
y_test_pred = svm_clf.predict(X_test)

# 4. Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}\n")

# 5. Classification Report
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred))


# 7. 3D Plot of Test Set Predictions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test_pred, cmap='viridis', marker='o')
legend1 = ax.legend(*scatter.legend_elements(), title="Predicted")
ax.add_artist(legend1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("3D SVM Predictions (Test Set)")
plt.show()

# 8. 3D Plot of Training Set Predictions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train_pred, cmap='viridis', marker='o')
legend1 = ax.legend(*scatter.legend_elements(), title="Predicted")
ax.add_artist(legend1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("3D SVM Predictions (Training Set)")
plt.show()
