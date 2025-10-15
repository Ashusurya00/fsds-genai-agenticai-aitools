import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

df = pd.read_csv(r'C:\Users\aashutosh\Downloads\Salary_Data.csv')

X = df.iloc[:,:-1].values
y = df.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

comparison = pd.DataFrame({'Accuracy':y_test, 'Predicted':y_pred})
print(comparison)

plt.scatter(X_train, y_train, color='red') 
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])
print(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f"predicted salary of 20 years of experience: ${y_20[0]:,.2f}")

bias = regressor.score(X_train, y_train)
variance = regressor.score(X_test, y_test)
train_mse = mean_squared_error(y_train, regressor.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f'training score (R^2):{bias:.2f}')
print(f'testing score (r^2): {variance:.2f}')
print(f'training mse: {train_mse:.2f}')
print(f'Test mse: {test_mse:.2f}')

'''filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print('model has been picked and saved as linear_regression_model.pkl')

import os
print(os.getcwd())'''

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

print('r-square:', r2)
print('mse:', mse)
print('rmse:', rmse)

'''from statsmodels.api import ols
ols(y_train, X_train).fit().summary()'''
import statsmodels.formula.api as smf

# Formula-style regression
model = smf.ols("Salary ~ YearsExperience", data=df).fit()
print(model.summary())
