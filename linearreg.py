import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- Sample dataset ----------------
data = {
    'Experience': [1,2,3,4,5,6,7,8,9,10],   # Independent variable (X)
    'Salary': [30000,35000,40000,45000,50000,
               60000,65000,70000,75000,80000]  # Dependent variable (y)
}
df = pd.DataFrame(data)

# ---------------- Features & Target ----------------
X = df[['Experience']]   # X must be a 2D array (features)
y = df['Salary']         # y is the target

# ---------------- Train-test split ----------------
# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Model Training ----------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- Predictions ----------------
y_pred = model.predict(X_test)

# ---------------- Evaluation ----------------
print("Intercept:", model.intercept_)   # Base salary (when experience = 0)
print("Coefficient:", model.coef_)      # Increase in salary per year of experience
print("MSE:", mean_squared_error(y_test, y_pred))  # Mean Squared Error
print("R2 Score:", r2_score(y_test, y_pred))       # Goodness of fit (closer to 1 is better)

# ---------------- Visualization ----------------
plt.scatter(X, y, color='blue', label="Data Points")     # Actual data
plt.plot(X, model.predict(X), color='red', label="Regression Line")  # Best fit line
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.legend()
plt.show()


