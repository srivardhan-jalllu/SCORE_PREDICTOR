import streamlit as st
from cricket_predictor import train_model, predict_score

st.set_page_config(page_title="Cricket Score Predictor")

st.title("🏏 Cricket Score Predictor")
st.write("Predict final cricket score using Machine Learning")

# Train model once
@st.cache_resource
def load_model():
    model, scaler, _ = train_model('ipl.csv')
    return model, scaler

model, scaler = load_model()

st.header("Enter Current Match State")

runs = st.number_input("Current Runs", min_value=0)
wickets = st.number_input("Wickets Lost", min_value=0, max_value=10)
overs = st.number_input("Overs Completed", min_value=0.0)
striker = st.number_input("Striker Runs", min_value=0)
non_striker = st.number_input("Non-Striker Runs", min_value=0)

if st.button("Predict Final Score"):
    prediction = predict_score(model, scaler, runs, wickets, overs, striker, non_striker)
    st.success(f"🎯 Predicted Final Score: {int(prediction)} runs")