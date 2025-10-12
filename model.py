# ===============================================================
# 🍏 AVOCADO PRICE PREDICTION — MULTIPLE MODEL TRAINING
# Saves: best_model.pkl, scaler.pkl, model_columns.pkl
# ===============================================================

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
data = pd.read_csv(r'C:\Users\aashutosh\OneDrive\Desktop\machine learning notes\8th - REGRESSION PROJECT\RESUME PROJECT -- PRICE PREDICTION\avocado.csv', index_col=0)

# Encode categorical type
data['type'] = data['type'].map({'conventional':0,'organic':1})
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data.drop('Date', axis=1, inplace=True)

# Map month numbers to names
month_map = {1:'JAN',2:'FEB',3:'MARCH',4:'APRIL',5:'MAY',6:'JUNE',
             7:'JULY',8:'AUG',9:'SEPT',10:'OCT',11:'NOV',12:'DEC'}
data['Month'] = data['Month'].map(month_map)

# Create dummy variables for year, region, Month
dummies = pd.get_dummies(data[['year','region','Month']], drop_first=True)

# Final dataset
X = pd.concat([
    data[['Total Volume','4046','4225','4770','Total Bags','Small Bags','Large Bags','XLarge Bags','type']],
    dummies
], axis=1)

y = data['AveragePrice']

# Save model columns for Flask
pickle.dump(X.columns, open('model_columns.pkl', 'wb'))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numeric features
numeric_cols = ['Total Volume','4046','4225','4770','Total Bags','Small Bags','Large Bags','XLarge Bags']
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'KNN Regressor': KNeighborsRegressor(n_neighbors=5),
    'SVR': SVR(gamma='scale'),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

# Evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        'MAE': round(mean_absolute_error(y_test, preds), 3),
        'MSE': round(mean_squared_error(y_test, preds), 3),
        'R2': round(r2_score(y_test, preds), 3)
    }

# Select best model based on R2 score
best_model_name = max(results, key=lambda k: results[k]['R2'])
best_model = models[best_model_name]

# Refit on entire training data
best_model.fit(X_train, y_train)

# Save best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Print results
print(f"✅ Best Model: {best_model_name}")
print("✅ Best model saved as 'best_model.pkl'")
print("\nAll model performances:")
for name, metrics in results.items():
    print(f"{name}: {metrics}")
