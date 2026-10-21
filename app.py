import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration
st.set_page_config(
    page_title="Loan Eligibility Prediction App",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Eligibility Prediction System")
st.write("Fill in the details below to check loan approval status.")

# Load the trained model
@st.cache_resource
def load_model():
    # Make sure to update 'model.pkl' with the actual path/name of your saved file
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model file: {e}")
    st.stop()

st.markdown("---")

# User Input Form
with st.form(key="loan_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        married = st.selectbox("Married", options=["Yes", "No"])
        dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])
        education = st.selectbox("Education", options=["Graduate", "Not Graduate"])

    with col2:
        self_employed = st.selectbox("Self Employed", options=["No", "Yes"])
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0, step=500)
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150, step=10)

    with col3:
        loan_term = st.selectbox("Loan Amount Term (in days/months)", options=[360, 180, 240, 120, 84, 60])
        credit_history = st.selectbox("Credit History (1.0 = Good, 0.0 = Poor)", options=[1.0, 0.0])
        property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])

    submit_button = st.form_submit_button(label="Predict Loan Status")

if submit_button:
    # Categorical Encoding Map (Adjust if your training preprocessing was different)
    gender_map = {"Male": 1, "Female": 0}
    married_map = {"Yes": 1, "No": 0}
    dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    education_map = {"Graduate": 1, "Not Graduate": 0}
    self_employed_map = {"Yes": 1, "No": 0}
    property_area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}

    # Construct input dataframe matching exact feature names required by model
    input_data = pd.DataFrame([{
        "Gender": gender_map[gender],
        "Married": married_map[married],
        "Dependents": dependents_map[dependents],
        "Education": education_map[education],
        "Self_Employed": self_employed_map[self_employed],
        "ApplicantIncome": float(applicant_income),
        "CoapplicantIncome": float(coapplicant_income),
        "LoanAmount": float(loan_amount),
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": float(credit_history),
        "Property_Area": property_area_map[property_area]
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.markdown("### Prediction Results")
    if prediction == 1:
        st.success(f"🎉 **Loan Approved!** (Confidence: {prediction_proba[1]*100:.2f}%)")
    else:
        st.error(f"❌ **Loan Rejected.** (Confidence: {prediction_proba[0]*100:.2f}%)")
