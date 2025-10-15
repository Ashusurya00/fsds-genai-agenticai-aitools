import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load the Breast Cancer dataset from sklearn
data = load_breast_cancer()

# Convert the dataset into a pandas DataFrame for easier handling
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add the target column (0 = malignant, 1 = benign)
df['target'] = data.target

# Display first 5 rows of the dataset
df.head()

# Count how many samples belong to each class
df['target'].value_counts()

# Features (first 30 columns)
X = df.iloc[:, 0:30]

# Target variable (last column)
y = df.iloc[:, -1]

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Create a Logistic Regression model
model = LogisticRegression(max_iter=5000)  # max_iter added to avoid convergence warnings
model.fit(X_train, y_train)  # Train the model

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
print('accuracy:', accuracy_score(y_pred, y_test))

from sklearn.metrics import confusion_matrix, classification_report

# Generate Confusion Matrix (shows TP, FP, TN, FN)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Generate classification report (Precision, Recall, F1-score, Support)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Get feature importance from logistic regression coefficients
importance = pd.Series(model.coef_[0], index=X.columns)

# Sort importance values in descending order
importance = importance.sort_values(ascending=False)

# Print top 10 most important features
print(importance.head(10))

# Plot top features as a bar chart
importance.plot(kind='bar', figsize=(12,5))
plt.title("Feature Importance in Logistic Regression")
plt.show()

from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for class 1 (benign)
y_prob = model.predict_proba(X_test)[:, 1]

# Compute False Positive Rate, True Positive Rate for ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line (random guess baseline)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()




