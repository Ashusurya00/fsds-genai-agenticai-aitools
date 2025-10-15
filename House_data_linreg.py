import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

dataset = pd.read_csv(r'C:\Users\aashutosh\OneDrive\Desktop\mlprac\House_data.csv')
space = dataset['sqft_living']
price = dataset['price']

x = np.array(space).reshape(-1,1)
y = np.array(price).reshape(-1,1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
dataset = dataset.drop(columns=['id','date'])

x = dataset.drop(columns=['price'])
y = dataset['price']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test, y_pred)

print('Multiple Linear Regression Performance')
print('R^2:',r2)
print('rmse:', rmse)
print('mae:',mae)

coeff_dataset = pd.DataFrame(regressor.coef_, x.columns, columns=['Coefficient'])
print('\nFeature coefficients:\n', coeff_dataset)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5,color='blue')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted prices (multiple Linear regression)')
plt.show()


