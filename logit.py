import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

data = pd.read_csv(r'C:\Users\aashutosh\Downloads/logit classification.csv')

X = data.iloc[:, [2, 3]].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

bias = classifier.score(X_train, y_train)
print(bias)

variance = classifier.score(X_test, y_test)
print(variance)

dataset1 = pd.read_csv(r"C:\Users\aashutosh\OneDrive\Attachments\Desktop\machine learning notes\2.LOGISTIC REGRESSION CODE\final1.csv")
d2 = dataset1.copy()

dataset1 = dataset1.iloc[:,[3,4]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

M = sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()

d2 ['y_pred1'] = classifier.predict(M)

d2.to_csv('final1.csv')
import os
os.getcwd()


from sklearn.svm import SVC

X_train1 = data.iloc[:,[2,3]].values
y_train1 = data.iloc[:-1].values

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=0)

classifier_svm = SVC()
classifier_svm.fit(X_train1, y_train1)

y_pred_svm = classifier_svm.predict(X_test1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_pred_svm)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test1, y_pred_svm)
print(ac)

from sklearn.metrics import classification_report
cr = classification_report(y_test1, y_pred_svm)
print(cr)

bias = classifier.score(X_train1, y_train1)
print(bias)


variance = classifier.score(X_test1, y_test1)
print(variance)

dataset2 = pd.read_csv(r"C:\Users\aashutosh\OneDrive\Attachments\Desktop\machine learning notes\2.LOGISTIC REGRESSION CODE\final1.csv")
d3 = dataset2.copy()

dataset2 = dataset2.iloc[:,[3,4]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

M = sc.fit_transform(dataset2)

y_pred_svm = pd.DataFrame()

d3 ['y_pred_svm'] = classifier_svm.predict(M)

d3.to_csv('final2.csv')
import os
os.getcwd()
