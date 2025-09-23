import streamlit as st
import pickle 
import numpy as np

model = pickle.load(open(r'C:\Users\aashutosh\OneDrive\Desktop\mlprac\linear_regression_model.pkl', 'rb'))

st.title('Salary Prediction App')

st.write('This app predicts the salary based on years of experience using a simple linear regresssion model.')

years_experience = st.number_input('Enter Years of Experienced:', min_value = 0.0, max_value=50.0, value=1.0, step=0.5)


if st.button('Predict salary'):
    experience_input = np.array([[years_experience]])
    prediction = model.predict(experience_input)

    st.success(f'The predicted salary for {years_experience} years of experience is: ${prediction[0]:,.2f}')


st.write('The model was trained using a dataset of salaries and years of experience') 

