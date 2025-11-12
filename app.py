import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load trained model
# -------------------------------
MODEL_FILE = "rf_model.pkl"

try:
    model = joblib.load(MODEL_FILE)
    st.success(f"✅ Model loaded successfully from {MODEL_FILE}")
except Exception as e:
    st.error(f"❌ Could not load model file ({MODEL_FILE}). Error: {e}")
    model = None

# -------------------------------
# App title
# -------------------------------
st.title("🧠 Employee Attrition Prediction")
st.write("Enter employee details below to predict attrition risk.")

# -------------------------------
# Input features
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    monthly_income = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=5000)
    age = st.slider("Age", 18, 70, 30)
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=60, value=8)
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    daily_rate = st.number_input("Daily Rate", min_value=0, max_value=20000, value=800)

with col2:
    years_at_company = st.number_input("Years At Company", min_value=0, max_value=60, value=5)
    hourly_rate = st.number_input("Hourly Rate", min_value=0, max_value=1000, value=50)
    distance_from_home = st.number_input("Distance From Home", min_value=0, max_value=100, value=5)
    monthly_rate = st.number_input("Monthly Rate", min_value=0, max_value=300000, value=20000)
    num_companies_worked = st.number_input("Num Companies Worked", min_value=0, max_value=50, value=2)

# Convert categorical
overtime_yes = 1 if overtime == "Yes" else 0

# Build DataFrame
input_data = pd.DataFrame({
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

# -------------------------------
# Predict
# -------------------------------
if st.button("🔍 Predict Attrition"):
    if model is None:
        st.error("Model not loaded. Make sure 'rf_model.pkl' is in the same folder as app.py.")
    else:
        prediction = model.predict(input_data)[0]
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
        else:
            probability = None

        if prediction == 1:
            st.error(f"⚠️ High risk of attrition{' (Prob: %.2f)' % probability if probability is not None else ''}")
        else:
            st.success(f"✅ Low risk of attrition{' (Prob: %.2f)' % probability if probability is not None else ''}")
