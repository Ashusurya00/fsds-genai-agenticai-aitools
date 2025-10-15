#importin libraries which are required
import numpy as np
import pandas as pd

#from sklearn importing modules and functions which are required
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

#getting dataset from local storage or downloaded from internet
df = pd.read_csv(r'C:\Users\aashutosh\Downloads\Data (2).csv')


#giving or assigning the columns to the X = independted var and y = dependent
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

#imputation of NaN values
imputer = SimpleImputer(strategy='mean')

X[:,1:3] = imputer.fit_transform(X[:,1:3])


#Label encoding of Categorical values
label_encoder = LabelEncoder()

X[:,0] = label_encoder.fit_transform(X[:,0])

y = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=0)

model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)


y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_pred,y_test))