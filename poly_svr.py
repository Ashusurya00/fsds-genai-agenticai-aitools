import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Users\aashutosh\Downloads\emp_sal.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(fit_intercept=False, n_jobs=2)
lin_reg.fit(X,y)

lin_model_pred = lin_reg.predict([[6.5]])
print(lin_model_pred)

plt.scatter(X,y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Linear Regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

plt.scatter(X,y , color='red')
plt.plot(X, lin_reg_2.predict((poly_reg).fit_transform(X)), color='blue')
plt.title('TRuth or bluff')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

lin_model_pred = lin_reg.predict([[6.5]])
print(lin_model_pred)

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)

from sklearn.svm import SVR
svr_model = SVR(kernel='poly', degree=4, gamma='auto', C= 10.0)
svr_model.fit(X,y)

svr_model_pred = svr_model.predict([[6.5]])
print(svr_model_pred)

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=5,weights='distance', algorithm='brute', p=1)
knn_model.fit(X,y)

knn_model_pred = knn_model.predict([[6.5]])
print(knn_model_pred)

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X,y)

dt_model_pred = dt_model.predict([[6.5]])
print(dt_model_pred)

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=23,random_state=0)
rf_model.fit(X,y)
rf_model_pred = rf_model.predict([[6.5]])
print(rf_model_pred)

