import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(page_title='Employee Attrition Prediction', layout='centered')
st.title('🧠 Employee Attrition Prediction')

MODEL_FILE = 'rf_model.pkl'
FEATURE_FILE = 'feature_columns.json'

# Load model
model, feature_cols = None, None
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    st.success('✅ Model loaded successfully')
else:
    st.error('❌ Model file not found')

# Load feature list
if os.path.exists(FEATURE_FILE):
    with open(FEATURE_FILE, 'r') as f:
        feature_cols = json.load(f)
else:
    st.error('❌ Feature list not found')

# Input section
st.header('Enter Employee Details')
col1, col2 = st.columns(2)
with col1:
    MonthlyIncome = st.number_input('Monthly Income', value=5000)
    Age = st.slider('Age', 18, 70, 30)
    TotalWorkingYears = st.number_input('Total Working Years', value=8)
    OverTime_Yes = st.selectbox('OverTime', ['No', 'Yes'])
    DailyRate = st.number_input('Daily Rate', value=800)
with col2:
    YearsAtCompany = st.number_input('Years At Company', value=5)
    HourlyRate = st.number_input('Hourly Rate', value=50)
    DistanceFromHome = st.number_input('Distance From Home', value=5)
    MonthlyRate = st.number_input('Monthly Rate', value=20000)
    NumCompaniesWorked = st.number_input('Num Companies Worked', value=2)

OverTime_Yes = 1 if OverTime_Yes == 'Yes' else 0

input_df = pd.DataFrame([{
    'MonthlyIncome': MonthlyIncome,
    'Age': Age,
    'TotalWorkingYears': TotalWorkingYears,
    'OverTime_Yes': OverTime_Yes,
    'DailyRate': DailyRate,
    'YearsAtCompany': YearsAtCompany,
    'HourlyRate': HourlyRate,
    'DistanceFromHome': DistanceFromHome,
    'MonthlyRate': MonthlyRate,
    'NumCompaniesWorked': NumCompaniesWorked
}])

st.subheader('Input Preview')
st.dataframe(input_df)

if st.button('🔍 Predict Attrition'):
    if model is None or feature_cols is None:
        st.error('Model or feature list not loaded.')
    else:
        X_pred = input_df.copy()
        for c in feature_cols:
            if c not in X_pred.columns:
                X_pred[c] = 0
        X_pred = X_pred[feature_cols]
        pred = model.predict(X_pred)[0]
        prob = None
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_pred)[0][1]
        if pred == 1:
            st.error(f'⚠️ High risk of attrition (Prob: {prob:.2f})')
        else:
            st.success(f'✅ Low risk of attrition (Prob: {prob:.2f})')
