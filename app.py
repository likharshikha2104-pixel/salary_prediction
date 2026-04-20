import streamlit as st
import joblib
import pandas as pd
import os

# Load model safely
@st.cache_resource
def load_model():
    model_path = os.path.join('saved_models', 'random_forest_model.pkl')
    
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please check deployment.")
        st.stop()
    
    return joblib.load(model_path)

model = load_model()

# UI
st.title("💰 Salary Prediction App")

st.write("Enter details to predict salary")

# Inputs
rating = st.slider('Rating (1-5)', 1.0, 5.0, 3.5)
salaries_reported = st.number_input('Salaries Reported', 1, 1000, 10)
company_name_encoded = st.number_input('Company Name Encoded', 0, 11039, 5000)

# Prediction
if st.button("Predict Salary"):
    input_data = pd.DataFrame([{
        'Rating': rating,
        'Salaries Reported': salaries_reported,
        'Company Name Encoded': company_name_encoded
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")
