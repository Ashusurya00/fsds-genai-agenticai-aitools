
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# =========================
# 1️⃣ Load the main dataset
# =========================
data = pd.read_csv(r'C:\Users\aashutosh\Downloads\logit classification.csv')
    
X = data.iloc[:, [2, 3]].values   # features
y = data.iloc[:, -1].values       # target

# =========================
# 2️⃣ Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# =========================
# 3️⃣ Standardization
# =========================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =========================
# 4️⃣ Logistic Regression Model
# =========================
classifier_log = LogisticRegression()
classifier_log.fit(X_train, y_train)

y_pred_log = classifier_log.predict(X_test)

# Evaluation
print("=== Logistic Regression Results ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))
print("Bias (Train Score):", classifier_log.score(X_train, y_train))
print("Variance (Test Score):", classifier_log.score(X_test, y_test))
print("\n")

# =========================
# 5️⃣ Save predictions on another dataset
# =========================
dataset1 = pd.read_csv(r"C:\Users\aashutosh\OneDrive\Attachments\Desktop\machine learning notes\2.LOGISTIC REGRESSION CODE\final1.csv")
d2 = dataset1.copy()

M = sc.transform(dataset1.iloc[:, [3, 4]].values)
d2['y_pred_log'] = classifier_log.predict(M)
d2.to_csv('final1_output.csv', index=False)
print("✅ Logistic Regression predictions saved to 'final1_output.csv'")

# =========================q
# 6️⃣ Support Vector Machine Model
# =========================
classifier_svm = SVC(kernel='rbf', random_state=0)
classifier_svm.fit(X_train, y_train)

y_pred_svm = classifier_svm.predict(X_test)

print("\n=== SVM Results ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Bias (Train Score):", classifier_svm.score(X_train, y_train))
print("Variance (Test Score):", classifier_svm.score(X_test, y_test))

# =========================
# 7️⃣ Save SVM predictions on same dataset
# =========================
dataset2 = pd.read_csv(r"C:\Users\aashutosh\OneDrive\Attachments\Desktop\machine learning notes\2.LOGISTIC REGRESSION CODE\final1.csv")
d3 = dataset2.copy()

M2 = sc.transform(dataset2.iloc[:, [3, 4]].values)
d3['y_pred_svm'] = classifier_svm.predict(M2)
d3.to_csv('final2_output.csv', index=False)
print("✅ SVM predictions saved to 'final2_output.csv'")

