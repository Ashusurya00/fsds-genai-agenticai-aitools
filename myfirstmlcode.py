import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\aashutosh\Downloads\Data (2).csv")



X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X[:, 1:3]=imputer.fit_transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X[:,0] = label_encoder.fit_transform(X[:,0])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=0)

from sklearn.preprocessing import Normalizer

sc_X = Normalizer()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

