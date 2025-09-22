import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)



df['target'] = data.target
df.head()
df['target'].value_counts()

X = df.iloc[:,0:30]

y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('accuracy:', accuracy_score(y_pred, y_test))

from sklearn.metrics import confusion_matrix, classification_report


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


print("\nClassification Report:\n", classification_report(y_test, y_pred))


importance = pd.Series(model.coef_[0], index=X.columns)


importance = importance.sort_values(ascending=False)

print(importance.head(10))


importance.plot(kind='bar', figsize=(12,5))
plt.title("Feature Importance in Logistic Regression")
plt.show()


from sklearn.metrics import roc_curve, auc


y_prob = model.predict_proba(X_test)[:,1]


fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.2f)' % roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()





