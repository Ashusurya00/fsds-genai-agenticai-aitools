import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv(r'C:\Users\aashutosh\OneDrive\Desktop\mlprac\House_data.csv')

# Extract two columns for Simple Linear Regression
space = dataset['sqft_living']   # Independent variable
price = dataset['price']         # Dependent variable

# Convert into numpy arrays and reshape into column vectors
x = np.array(space).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)

# Split data into training and test sets (1/3 for testing)
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=1/3, random_state=0)

# Create and train Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# Predict on test set
pred = regressor.predict(xtest)

# ---------------- Visualizations ----------------
# Training set visualization
plt.scatter(xtrain, ytrain, color='red')   # Scatter plot of actual data
plt.plot(xtrain, regressor.predict(xtrain), color='blue')  # Regression line
plt.title('Visuals for Training set')
plt.xlabel('Space (sqft_living)')
plt.ylabel('Price')
plt.show()

# Test set visualization
plt.scatter(xtest, ytest, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title('Visuals for Test set')
plt.xlabel('Space (sqft_living)')
plt.ylabel('Price')
plt.show()

# (Duplicate visualization - same as above, you can remove one)
plt.scatter(xtest, ytest, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title('Visuals for Test set')
plt.xlabel('Space (sqft_living)')
plt.ylabel('Price')
plt.show()

# ---------------- Multiple Linear Regression ----------------
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Drop unnecessary columns that are not useful for regression
dataset = dataset.drop(columns=['id', 'date'])

# Define features (X) and target (y)
x = dataset.drop(columns=['price'])   # All independent variables
y = dataset['price']                  # Dependent variable

# Split dataset into training and testing sets (80/20 split)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create and train regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict on test data
y_pred = regressor.predict(x_test)

# ---------------- Model Evaluation ----------------
# R^2 score → how well the model explains variance
r2 = r2_score(y_test, y_pred)

# RMSE → average error in price units (root of mean squared error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# MAE → average absolute error in price prediction
mae = mean_absolute_error(y_test, y_pred)

print('Multiple Linear Regression Performance')
print('R^2:', r2)
print('RMSE:', rmse)
print('MAE:', mae)

# ---------------- Feature Coefficients ----------------
# Show how much each feature contributes to price
coeff_dataset = pd.DataFrame(regressor.coef_, x.columns, columns=['Coefficient'])
print('\nFeature coefficients:\n', coeff_dataset)

# ---------------- Visualization: Actual vs Predicted ----------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices (Multiple Linear Regression)')
plt.show()
