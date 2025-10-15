import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import statsmodels.formula.api as smf
import os

# ---------------- Load Dataset ----------------
df = pd.read_csv(r'C:\Users\aashutosh\Downloads\Salary_Data.csv')

# Features (Years of Experience) and Target (Salary)
X = df.iloc[:, :-1].values   # Independent variable
y = df.iloc[:, 1].values     # Dependent variable

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ---------------- Train Linear Regression ----------------
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# ---------------- Predictions ----------------
y_pred = regressor.predict(X_test)
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Actual vs Predicted:\n", comparison)

# ---------------- Visualizations ----------------
plt.scatter(X_train, y_train, color='red', label='Training Data')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Regression Line')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Regression Line')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# ---------------- Custom Predictions ----------------
pred_12 = regressor.predict([[12]])
pred_20 = regressor.predict([[20]])
print(f"Predicted salary for 12 years of experience: ${pred_12[0]:,.2f}")
print(f"Predicted salary for 20 years of experience: ${pred_20[0]:,.2f}")

# ---------------- Model Evaluation ----------------
train_r2 = regressor.score(X_train, y_train)
test_r2 = regressor.score(X_test, y_test)
train_mse = mean_squared_error(y_train, regressor.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(test_mse)

print(f"Training R²: {train_r2:.2f}")
print(f"Testing R²: {test_r2:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Testing MSE: {test_mse:.2f}")
print(f"Testing RMSE: {rmse:.2f}")
print(f"Intercept: {regressor.intercept_}")
print(f"Coefficient: {regressor.coef_[0]}")

# ---------------- Save Model using Pickle ----------------
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print(f'Model saved as {filename} in {os.getcwd()}')

# ---------------- Statsmodels OLS Summary ----------------
model_ols = smf.ols("Salary ~ YearsExperience", data=df).fit()
print("\nOLS Regression Summary:\n")
print(model_ols.summary())
