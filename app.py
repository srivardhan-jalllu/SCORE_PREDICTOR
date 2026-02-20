import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(
    page_title="🏏 Cricket Score Predictor",
    page_icon="🏏",
    layout="wide"
)

# Title
st.title("🏏 Cricket Score Predictor")
st.markdown("Predict cricket match scores using Machine Learning")

# Custom accuracy function
def custom_accuracy(y_test, y_pred, threshold):
    """Calculate custom accuracy based on threshold"""
    right = 0
    l = len(y_pred)
    for i in range(0, l):
        if abs(y_pred[i] - y_test[i]) <= threshold:
            right += 1
    return (right/l) * 100

# Train model function
@st.cache_resource
def train_model(csv_file='ipl.csv'):
    """Train the cricket score prediction model"""
    try:
        # Load dataset
        dataset = pd.read_csv(csv_file)
        
        # Extract features and labels
        X = dataset.iloc[:,[7,8,9,12,13]].values
        y = dataset.iloc[:, 14].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        
        # Feature scaling
        sc = StandardScaler()
        X_train_scaled = sc.fit_transform(X_train)
        X_test_scaled = sc.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, max_features=None, random_state=0)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        
        return model, sc, X_test, y_test, y_pred
    except FileNotFoundError:
        st.error(f"❌ Dataset '{csv_file}' not found. Please ensure the CSV file exists.")
        return None, None, None, None, None

def predict_score(model, scaler, runs, wickets, overs, striker, non_striker):
    """Predict score for given match state"""
    if model is None:
        return None
    input_data = np.array([[runs, wickets, overs, striker, non_striker]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]

# Sidebar - Dataset selection
st.sidebar.title("⚙️ Configuration")
dataset_choice = st.sidebar.selectbox(
    "Select Dataset:",
    ["IPL", "ODI", "T20"],
    help="Choose the cricket format dataset"
)

dataset_map = {
    "IPL": "ipl.csv",
    "ODI": "odi.csv",
    "T20": "t20.csv"
}

csv_file = dataset_map[dataset_choice]

# Train the model
with st.spinner("🔄 Training model..."):
    model, scaler, X_test, y_test, y_pred = train_model(csv_file)

if model is not None:
    # Display metrics
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Performance")
    r2_score = model.score(scaler.transform(X_test), y_test) * 100
    acc_10 = custom_accuracy(y_test, y_pred, 10)
    acc_20 = custom_accuracy(y_test, y_pred, 20)
    
    st.sidebar.metric("R-squared Score", f"{r2_score:.2f}%")
    st.sidebar.metric("Accuracy (±10 runs)", f"{acc_10:.2f}%")
    st.sidebar.metric("Accuracy (±20 runs)", f"{acc_20:.2f}%")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Enter Match State")
        runs = st.number_input("Current Runs", min_value=0, max_value=250, value=100)
        wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=2)
        overs = st.number_input("Overs Bowled", min_value=0.0, max_value=20.0, value=12.0)
    
    with col2:
        st.subheader("🏏 Batsmen Performance")
        striker = st.number_input("Striker Runs", min_value=0, max_value=150, value=50)
        non_striker = st.number_input("Non-striker Runs", min_value=0, max_value=150, value=40)
    
    # Make prediction
    if st.button("🎯 Predict Final Score", use_container_width=True):
        predicted = predict_score(model, scaler, runs, wickets, overs, striker, non_striker)
        
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Final Score", f"{predicted:.0f} runs")
        with col2:
            mean_error = np.mean(y_pred - y_test)
            st.metric("Average Model Error", f"±{np.std(y_pred - y_test):.0f} runs")
        
        # Show confidence
        st.info(
            f"📌 Based on {len(y_test):,} test samples, the model predicts "
            f"**{predicted:.0f} runs** as the final score with typical error margin of ±{np.std(y_pred - y_test):.0f} runs."
        )
    
    # Example predictions
    st.markdown("---")
    st.subheader("📋 Example Predictions")
    
    examples = [
        (100, 0, 13, 50, 50, "Good start, no wickets"),
        (60, 3, 10, 30, 20, "Moderate start, some wickets"),
        (40, 5, 8, 15, 15, "Slow start, many wickets"),
        (150, 1, 15, 80, 60, "Great position"),
    ]
    
    cols = st.columns(2)
    for idx, (r, w, o, s, ns, desc) in enumerate(examples):
        pred = predict_score(model, scaler, r, w, o, s, ns)
        with cols[idx % 2]:
            st.write(f"**{desc}**")
            st.write(f"• Runs: {r} | Wickets: {w} | Overs: {o}")
            st.write(f"• Striker: {s} | Non-striker: {ns}")
            st.success(f"➜ Predicted: **{pred:.0f} runs**")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
        <p>🏏 Cricket Score Predictor | Using Random Forest Regression</p>
        <p style='font-size: 12px; color: gray;'>Built with Streamlit | Dataset: IPL, ODI, T20</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.error("❌ Could not load the model. Please check that all CSV files are present.")
