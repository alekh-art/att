
# app.py
"""
Streamlit app — Employee Attrition Prediction
This version is ready to download from Colab and then upload to your GitHub repo.
It checks several sensible locations for rf_model.pkl (including model/rf_model.pkl
or rf_model.pkl placed in the same folder as app.py when deployed on Streamlit).
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")
st.title("🧠 Employee Attrition Prediction")
st.markdown("Enter the employee's details below and click **Predict** to see attrition risk.")

# Candidate model locations (Streamlit runs from repo root on deploy)
CANDIDATE_MODELS = [
    os.path.join("model", "rf_model.pkl"),        # model/rf_model.pkl (recommended)
    os.path.join(os.getcwd(), "rf_model.pkl"),    # ./rf_model.pkl (same folder as app.py)
    os.path.join("/content/drive/My Drive/MLMA DT", "rf_model.pkl"),  # optional Drive root fallback
    os.path.join("/content/drive/My Drive/MLMA DT/streamlit-attrition-app", "rf_model.pkl"),
]

MODEL_PATH = None
for p in CANDIDATE_MODELS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

rf_clf = None
if MODEL_PATH:
    try:
        rf_clf = joblib.load(MODEL_PATH)
        st.success(f"✅ Model loaded from: {MODEL_PATH}")
    except Exception as e:
        rf_clf = None
        st.error(f"❌ Failed to load model at {MODEL_PATH}: {e}")
else:
    st.warning(
        "⚠️ Model file not found. Expected at one of:\n" +
        "\n".join([f" - {p}" for p in CANDIDATE_MODELS]) +
        "\n\nPlace 'rf_model.pkl' in the repo (model/rf_model.pkl) or same folder as app.py and redeploy."
    )

# Input widgets for the top 10 features
st.header("Input Employee Details")
col1, col2 = st.columns(2)
with col1:
    monthly_income = st.number_input("Monthly Income", min_value=0, max_value=1_000_000, value=5000, step=100)
    age = st.slider("Age", 18, 70, 30)
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=60, value=8, step=1)
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    daily_rate = st.number_input("Daily Rate", min_value=0, max_value=20000, value=800, step=1)

with col2:
    years_at_company = st.number_input("Years At Company", min_value=0, max_value=60, value=5, step=1)
    hourly_rate = st.number_input("Hourly Rate", min_value=0, max_value=1000, value=50, step=1)
    distance_from_home = st.number_input("Distance From Home", min_value=0, max_value=100, value=5, step=1)
    monthly_rate = st.number_input("Monthly Rate", min_value=0, max_value=300000, value=20000, step=100)
    num_companies_worked = st.number_input("Num Companies Worked", min_value=0, max_value=50, value=2, step=1)

overtime_yes = 1 if overtime == "Yes" else 0

# Build input DataFrame in model-expected order
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

# Prediction
if rf_clf is not None:
    if st.button("🔍 Predict Attrition"):
        try:
            pred = rf_clf.predict(input_df)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            pred = None

        prob_text = ""
        try:
            if hasattr(rf_clf, "predict_proba"):
                proba = rf_clf.predict_proba(input_df)[0]
                if len(proba) > 1:
                    prob_text = f" (probability: {proba[1]:.2f})"
        except Exception:
            prob_text = ""

        if pred is None:
            pass
        elif pred == 1 or str(pred).lower() in ("yes", "true", "1"):
            st.error(f"⚠️ High risk of attrition{prob_text}")
        else:
            st.success(f"✅ Low risk of attrition{prob_text}")
else:
    st.info("Place 'rf_model.pkl' next to app.py or under model/rf_model.pkl in the repo and redeploy.")

st.markdown("---")
st.caption("Built with ❤️ — joblib + Streamlit")
