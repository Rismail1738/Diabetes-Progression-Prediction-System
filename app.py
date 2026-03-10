import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("linear_regression_model.pkl", "rb"))

# Page layout
st.set_page_config(page_title="Diabetes Progression Predictor", layout="wide")

# -----------------------------
# Title and Description
# -----------------------------
st.title("Diabetes Disease Progression Predictor")

st.write("""
This application predicts **diabetes disease progression** one year after baseline using a 
**Linear Regression model** trained on the Diabetes Dataset from scikit-learn.
""")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Patient Medical Data")

age = st.sidebar.number_input("Age (standardized)", value=0.0, format="%.2f")
sex = st.sidebar.number_input("Sex (standardized)", value=0.0, format="%.2f")
bmi = st.sidebar.number_input("Body Mass Index (BMI)", value=0.0, format="%.2f")
bp = st.sidebar.number_input("Blood Pressure", value=0.0, format="%.2f")
s1 = st.sidebar.number_input("Serum Measurement 1", value=0.0, format="%.2f")
s2 = st.sidebar.number_input("Serum Measurement 2", value=0.0, format="%.2f")
s3 = st.sidebar.number_input("Serum Measurement 3", value=0.0, format="%.2f")
s4 = st.sidebar.number_input("Serum Measurement 4", value=0.0, format="%.2f")
s5 = st.sidebar.number_input("Serum Measurement 5", value=0.0, format="%.2f")
s6 = st.sidebar.number_input("Serum Measurement 6", value=0.0, format="%.2f")

# Prepare input for prediction
features = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("Prediction")

if st.button("Predict Diabetes Progression"):

    prediction = model.predict(features)[0]

    # Classify risk level
    if prediction < 120:
        risk = "Low Progression"
        color = "green"
        progress_val = min(prediction / 120, 1.0)
    elif prediction < 220:
        risk = "Moderate Progression"
        color = "orange"
        progress_val = min((prediction - 120) / 100, 1.0)
    else:
        risk = "High Progression"
        color = "red"
        progress_val = min((prediction - 220) / 130, 1.0)

    st.success(f"Predicted Diabetes Progression Score: {prediction:.2f}")
    st.warning(f"Risk Level: {risk}")

    # Display a progress bar
    st.progress(progress_val)

    st.info("""
    Higher scores indicate greater diabetes progression.  
    Risk categories:
    - Low: < 120
    - Moderate: 120–220
    - High: > 220
    """)

# -----------------------------
# Model Information
# -----------------------------
st.subheader("Model Information")
st.write("""
**Dataset:** Diabetes Dataset from scikit-learn  
**Target Variable:** Disease progression after one year  
**Algorithm:** Linear Regression  
**Purpose:** Estimate diabetes progression using patient medical data.
""")

# -----------------------------
# Use Cases
# -----------------------------
st.subheader("Possible Applications")
st.write("""
• Clinical decision support for doctors  
• Monitoring diabetes risk factors  
• Medical research on disease progression  
• Educational demonstration of ML in healthcare
""")