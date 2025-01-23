# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 04:11:17 2024

@author: Dell
"""

import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('C:/Users/Dell/OneDrive/Desktop/Multiple Disease Prediction/Diabetes/trained_model.sav', 'rb'))

# Creating a function for prediction
def diabetes_prediction(input_data):
    # Convert the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # Giving a title
    st.title('Diabetes Prediction Web App')
    
    # Creating input data fields
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age")
    
    # Code for prediction
    diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Diabetes Test Result'):
        input_data = [int(Pregnancies), int(Glucose), int(BloodPressure), int(SkinThickness), 
                      int(Insulin), float(BMI), float(DiabetesPedigreeFunction), int(Age)]
        
        # Get the prediction
        diagnosis = diabetes_prediction(input_data)
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
