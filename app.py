import streamlit as st
import joblib
import pandas as pd
import os

# Define the path to the saved models
# In a deployed environment, ensure this path is correct relative to where app.py is placed
model_path = 'saved_models/random_forest_model.pkl' # Adjust if your model is in a different location

# Load the trained model
@st.cache_resource # Cache the model loading to improve performance
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}. Please ensure it's in the correct directory.")
        st.stop()
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model(model_path)

# Streamlit App Title
st.title('Salary Prediction App')

st.write("Enter the details below to get a salary prediction:")

# Input fields for features
rating = st.slider('Rating (1-5)', min_value=1.0, max_value=5.0, value=3.5, step=0.1)
salaries_reported = st.number_input('Salaries Reported', min_value=1, value=10, step=1)

# For 'Company Name Encoded', you would ideally use the same LabelEncoder
# that was used during training. For a simple demo, we'll allow manual input.
# The max value observed in `df['Company Name Encoded']` is 11039, so let's use that as a sensible upper bound.
company_name_encoded = st.number_input('Company Name Encoded (0-11039)', min_value=0, max_value=11039, value=5000, step=1)


# Create a DataFrame for prediction
input_data = pd.DataFrame([{
    'Rating': rating,
    'Salaries Reported': salaries_reported,
    'Company Name Encoded': company_name_encoded
}])

# Make prediction
if st.button('Predict Salary'):
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Salary: ₹{prediction:,.2f}')
