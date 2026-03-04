import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import fetch_openml

# Load model
model = joblib.load("production_credit_model.pkl")

st.title("Credit Risk Prediction System")

st.write("Enter customer details to assess credit risk.")

# Load dataset only to get column names
data = fetch_openml(name='credit-g', version=1, as_frame=True)
df = data.frame.drop('class', axis=1)

input_data = {}

for column in df.columns:
    if df[column].dtype == 'object':
        input_data[column] = st.selectbox(column, df[column].unique())
    else:
        min_val = float(df[column].min())
        max_val = float(df[column].max())
        mean_val = float(df[column].mean())
        input_data[column] = st.slider(column, min_val, max_val, mean_val)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"High Risk Customer ⚠️")
    else:
        st.success(f"Low Risk Customer ✅")

    st.write(f"Default Probability: {probability:.2f}")