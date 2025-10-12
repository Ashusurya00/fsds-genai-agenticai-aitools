from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# ---------------------------
# LOAD SAVED FILES
# ---------------------------
model = pickle.load(open('best_model.pkl', 'rb'))       # Best ML model
scaler = joblib.load('scaler.pkl')                     # Scaler saved with joblib
model_columns = pickle.load(open('model_columns.pkl', 'rb'))  # Columns used during training

# ---------------------------
# HOME PAGE
# ---------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------------------
# ABOUT PAGE
# ---------------------------
@app.route('/about')
def about():
    return render_template('about.html')

# ---------------------------
# PREDICTION
# ---------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs from form
        total_volume = float(request.form['total_volume'])
        total_4046 = float(request.form['4046'])
        total_4225 = float(request.form['4225'])
        total_4770 = float(request.form['4770'])
        total_bags = float(request.form['total_bags'])
        small_bags = float(request.form['small_bags'])
        large_bags = float(request.form['large_bags'])
        xlarge_bags = float(request.form['xlarge_bags'])
        year = int(request.form['year'])
        region = int(request.form['region'])
        avocado_type = int(request.form['type'])

        # Create dataframe with proper columns
        input_dict = {
            'Total Volume': total_volume,
            '4046': total_4046,
            '4225': total_4225,
            '4770': total_4770,
            'Total Bags': total_bags,
            'Small Bags': small_bags,
            'Large Bags': large_bags,
            'XLarge Bags': xlarge_bags,
            'type': avocado_type,
            'year': year,
            'region': region
        }

        input_df = pd.DataFrame([input_dict])

        # Add missing dummy columns with 0
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns as per training
        input_df = input_df[model_columns]

        # Scale numeric features
        numeric_cols = ['Total Volume','4046','4225','4770','Total Bags','Small Bags','Large Bags','XLarge Bags']
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

# ---------------------------
# RUN APP
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)
