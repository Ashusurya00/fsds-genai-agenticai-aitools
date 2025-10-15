import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Users\aashutosh\Downloads\emp_sal.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

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


print("SVR Prediction for 6.5:", svr_model_pred)

# Plotting SVR results
X_range = np.linspace(min(X), max(X), 200).reshape(-1, 1)
y_range = svr_model.predict(X_range)

plt.scatter(X, y, color='red', label='Data points')
plt.plot(X_range, y_range, color='blue', label='SVR Polynomial fit')
plt.scatter([6.5], svr_model_pred, color='green', label=f'Prediction at 6.5: {svr_model_pred[0]:.2f}')
plt.title('SVR Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend()
plt.show()




