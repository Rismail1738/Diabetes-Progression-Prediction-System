import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("linear_regression_model.pkl", "rb"))

# Title
st.title("Diabetes Progression Prediction System")

st.write("""
This application predicts diabetes disease progression using a Linear Regression model trained on the Diabetes Dataset from scikit-learn.
""")

# -------------------------
# Model Description
# -------------------------

st.header("Model Description")

st.write("""
Dataset Used: Diabetes Dataset from scikit-learn

Target Variable:
Disease progression one year after baseline.

Machine Learning Algorithm:
Linear Regression

Purpose:
To predict diabetes progression based on patient medical measurements.
""")

# -------------------------
# User Inputs
# -------------------------

st.header("Enter Patient Data")

age = st.slider("Age", -0.1, 0.1, 0.0)
sex = st.slider("Sex", -0.1, 0.1, 0.0)
bmi = st.slider("BMI", -0.1, 0.2, 0.0)
bp = st.slider("Blood Pressure", -0.1, 0.2, 0.0)
s1 = st.slider("S1", -0.2, 0.2, 0.0)
s2 = st.slider("S2", -0.2, 0.2, 0.0)
s3 = st.slider("S3", -0.2, 0.2, 0.0)
s4 = st.slider("S4", -0.2, 0.2, 0.0)
s5 = st.slider("S5", -0.2, 0.2, 0.0)
s6 = st.slider("S6", -0.2, 0.2, 0.0)

features = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])

# -------------------------
# Prediction
# -------------------------

if st.button("Predict"):
    
    prediction = model.predict(features)

    st.success(f"Predicted Diabetes Progression Score: {prediction[0]:.2f}")