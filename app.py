import streamlit as st
import joblib
import pandas as pd
import os

# -------------------- Load Model -------------------- #
@st.cache_resource
def load_model():
    model_path = "https://github.com/likharshikha2104-pixel/salary_prediction/blob/main/linear_regression_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Make sure it exists in 'saved_models/' folder.")
        st.stop()

    return joblib.load(model_path)

model = load_model()

# -------------------- UI -------------------- #
st.set_page_config(page_title="Salary Predictor", page_icon="💰")

st.title("💰 Salary Prediction App")
st.markdown("Enter the details below to predict salary")

# -------------------- Inputs -------------------- #
rating = st.slider("⭐ Rating", 1.0, 5.0, 3.5, step=0.1)
salaries_reported = st.number_input("📊 Salaries Reported", min_value=1, max_value=1000, value=10)
company_name_encoded = st.number_input("🏢 Company Code", min_value=0, max_value=11039, value=5000)

# -------------------- Prediction -------------------- #
if st.button("Predict Salary"):
    try:
        input_data = pd.DataFrame([{
            "Rating": rating,
            "Salaries Reported": salaries_reported,
            "Company Name Encoded": company_name_encoded
        }])

        prediction = model.predict(input_data)[0]

        st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
