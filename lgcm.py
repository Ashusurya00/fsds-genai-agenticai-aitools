import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv(r'C:\Users\aashutosh\Downloads\Data (2).csv')

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

imputer = SimpleImputer(strategy='mean')

X[:,1:3] = imputer.fit_transform(X[:,1:3])

label_encoded = LabelEncoder()
X[:,0] = label_encoded.fit_transform(X[:,0])

y = label_encoded.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.2, random_state=0)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

