# app.py
"""
Streamlit app — Employee Attrition Prediction
Loads a RandomForest (or similar sklearn) model saved with joblib from model/rf_model.pkl
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")
st.title("🧠 Employee Attrition Prediction")
st.markdown("Enter the employee's details below and click **Predict** to see attrition risk.")

# -----------------------------
# Load model from model/rf_model.pkl
# -----------------------------
MODEL_PATH = os.path.join("model", "rf_model.pkl")

if os.path.exists(MODEL_PATH):
    try:
        rf_clf = joblib.load(MODEL_PATH)
        st.success(f"✅ Model loaded from: {MODEL_PATH}")
    except Exception as e:
        rf_clf = None
        st.error(f"❌ Failed to load model: {e}")
else:
    rf_clf = None
    st.warning("⚠️ Model not found! Please place your trained model at `model/rf_model.pkl`.")

# -----------------------------
# Input features (Top 10)
# -----------------------------
st.header("Input Employee Details")

col1, col2 = st.columns(2)
with col1:
    monthly_income = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=5000, step=100)
    age = st.slider("Age", 18, 70, 30)
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=60, value=8)
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    daily_rate = st.number_input("Daily Rate", min_value=0, max_value=20000, value=800)

with col2:
    years_at_company = st.number_input("Years At Company", min_value=0, max_value=60, value=5)
    hourly_rate = st.number_input("Hourly Rate", min_value=0, max_value=1000, value=50)
    distance_from_home = st.number_input("Distance From Home", min_value=0, max_value=100, value=5)
    monthly_rate = st.number_input("Monthly Rate", min_value=0, max_value=300000, value=20000)
    num_companies_worked = st.number_input("Num Companies Worked", min_value=0, max_value=20, value=2)

# Encode OverTime
overtime_yes = 1 if overtime == "Yes" else 0

# Build dataframe in same order as model training
input_df = pd.DataFrame({
    "MonthlyIncome": [monthly_income],
    "Age": [age],
    "TotalWorkingYears": [total_working_years],
    "OverTime_Yes": [overtime_yes],
    "DailyRate": [daily_rate],
    "YearsAtCompany": [years_at_company],
    "HourlyRate": [hourly_rate],
    "DistanceFromHome": [distance_from_home],
    "MonthlyRate": [monthly_rate],
    "NumCompaniesWorked": [num_companies_worked]
})

st.markdown("---")
st.subheader("Input Summary")
st.dataframe(input_df)

# -----------------------------
# Predict
# -----------------------------
if rf_clf is not None:
    if st.button("🔍 Predict Attrition"):
        try:
            pred = rf_clf.predict(input_df)[0]
            proba = rf_clf.predict_proba(input_df)[0][1] if hasattr(rf_clf, "predict_proba") else None

            if pred == 1:
                st.error(f"⚠️ High risk of attrition{f' (Probability: {proba:.2f})' if proba else ''}")
            else:
                st.success(f"✅ Low risk of attrition{f' (Probability: {proba:.2f})' if proba else ''}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Upload your model file at `model/rf_model.pkl` and reload the app.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and joblib")
