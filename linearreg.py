import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    'Experience': [1,2,3,4,5,6,7,8,9,10],
    'Salary': [30000,35000,40000,45000,50000,60000,65000,70000,75000,80000]
}
df = pd.DataFrame(data)

# Features & Target
X = df[['Experience']]   # Independent variable
y = df['Salary']         # Dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization
plt.scatter(X, y, color='blue', label="Data Points")
plt.plot(X, model.predict(X), color='red', label="Regression Line")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.legend()
plt.show()

