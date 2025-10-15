
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

dataset = pd.read_csv(r'C:\Users\aashutosh\Downloads\logit classification.csv')

X = dataset.iloc[:, [2, 3]].values   # features
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier_knn= KNeighborsClassifier(n_neighbors=9)
classifier_knn.fit(X_train, y_train)

y_pred_knn = classifier_knn.predict(X_test)



print("=== Logistic Regression Results ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("Bias (Train Score):", classifier_knn.score(X_train, y_train))
print("Variance (Test Score):", classifier_knn.score(X_test, y_test))
print("\n")