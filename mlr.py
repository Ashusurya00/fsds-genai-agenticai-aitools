import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r'C:\Users\aashutosh\Downloads\Investment.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,4]

X = pd.get_dummies(X,dtype=int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

X = np.append(arr = np.full((50,1),42467).astype(int), values= X, axis=1)
 
import statsmodels.api as sm
X_OPT = X[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog=y, exog=X_OPT).fit()
regressor_OLS.summary()
print(regressor_OLS.summary())

import statsmodels.api as sm
X_OPT = X[:,[0,1,2,3,5]]

regressor_OLS = sm.OLS(endog=y, exog=X_OPT).fit()
regressor_OLS.summary()
print(regressor_OLS.summary())

import statsmodels.api as sm
X_OPT = X[:,[0,1,2,3]]

regressor_OLS = sm.OLS(endog=y, exog=X_OPT).fit()
regressor_OLS.summary()
print(regressor_OLS.summary())

import statsmodels.api as sm
X_OPT = X[:,[0,1,3]]

regressor_OLS = sm.OLS(endog=y, exog=X_OPT).fit()
regressor_OLS.summary()
print(regressor_OLS.summary())

import statsmodels.api as sm
X_OPT = X[:,[0,1]]

regressor_OLS = sm.OLS(endog=y, exog=X_OPT).fit()
regressor_OLS.summary()
print(regressor_OLS.summary())

bias = regressor.score(X_train, y_train)
bias

variance = regressor.score(X_test, y_test)
variance



