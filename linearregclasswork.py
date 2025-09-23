import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'C:\Users\aashutosh\Downloads\Salary_Data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

comparison = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparison)

plt.scatter(X_test,y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Experience vs Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

y_20 = m*20+c
print(y_20) 

bias = regressor.score(X_train, y_train)
print(bias)

variance = regressor.score(X_test, y_test)
print(variance)

df.mean()
df['Salary'].mean()

df.median()

df['Salary'].mode()

df.var()

df['Salary'].var()

df.std()

from scipy.stats import variation

variation(df.values)

variation(df['Salary'])

df.corr()
df['Salary'].corr(df['YearsExperience'])
df.skew()
df['Salary'].skew()
df.sem()
df['Salary'].sem()

import scipy.stats as stats
df.apply(stats.zscore)

stats.zscore(df['Salary'])

y_mean = np.mean(y)
ssr = np.sum((y_pred-y_mean)**2)
print(ssr)

y = y[0:6]
sse = np.sum((y-y_pred)**2)
print(sse)

mean_total = np.mean(df.values)
sst = np.sum((df.values-mean_total)**2)
print(ssr)

r_square = 1 - (ssr/sst)


from sklearn.metrics import mean_squared_error
train_mse = mean_squared_error(y_train, regressor.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)

print(train_mse)
print(test_mse)

#import pickle
#filename = 'linear_regression_model.pkl'