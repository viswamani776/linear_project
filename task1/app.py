import streamlit as st
import joblib
import numpy as np
import os

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "linear_model.pkl")

model = joblib.load(MODEL_PATH)

st.title("💼 Salary Prediction App")
st.write("Predict salary based on years of experience")

experience = st.number_input("Enter Years of Experience", min_value=0.0, step=0.1)

if st.button("Predict Salary"):
    prediction = model.predict([[experience]])
    st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")
