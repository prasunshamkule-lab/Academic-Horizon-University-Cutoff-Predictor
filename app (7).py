import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="University Admission Cutoff Predictor",
    page_icon="🎓",
    layout="centered"
)

# Title and Description
st.title("🎓 University Admission Cutoff Predictor")
st.write("Predict university admission cutoffs using this interactive machine learning model. Select your input parameters below to forecast cutoff values (Score or Rank).")

# Load the trained model and encoders safely
@st.cache_resource
def load_assets():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

try:
    model, encoders = load_assets()
    le_kind = encoders['le_kind']
    le_seat = encoders['le_seat']
    le_target = encoders['le_target']

    # User Inputs
    st.subheader("📍 Input Parameters")

    year = st.selectbox("Admission Year", [2024, 2025, 2026], index=1)
    figure_kind = st.selectbox("Figure Kind (Admission Metric)", list(le_kind.classes_))
    seat_type = st.selectbox("Seat Allocation Category", list(le_seat.classes_))
    basis = st.selectbox("Admission Basis", list(le_target.classes_))

    # Prediction Button
    if st.button("🔮 Predict Cutoff Figure"):
        # Encode user inputs using the pre-fitted encoders
        encoded_kind = le_kind.transform([figure_kind])[0]
        encoded_seat = le_seat.transform([seat_type])[0]
        encoded_basis = le_target.transform([basis])[0]

        # Build input feature vector
        # Feature format: ['year', 'figure_kind', 'seat_type', 'basis']
        features = np.array([[year, encoded_kind, encoded_seat, encoded_basis]])

        prediction = model.predict(features)[0]

        # Display results based on Figure Kind category
        st.success("🎉 Prediction Complete!")
        if figure_kind.lower() == 'score':
            st.metric(label="Predicted Cutoff Score", value=f"{prediction:.2f}")
        else:
            st.metric(label="Predicted Cutoff Rank", value=f"{int(np.round(prediction))}")

except FileNotFoundError:
    st.error("⚠️ model.pkl or encoders.pkl was not found. Please ensure they are uploaded in the same directory as app.py!")
