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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main styling */
    .main {
        padding: 2rem;
    }
    
    /* Card styling */
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        margin: 1rem 0;
    }
    
    /* Title styling */
    .title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: bold;
    }
    
    /* Metric boxes */
    .metric-box {
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>🏏 Cricket Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1em; color: #666;'>Predict cricket match scores using Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

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
        
        # Filter: Ignore data before 2015 (only use 2015 onwards)
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset = dataset[dataset['date'] >= '2015-01-01']
        
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
st.sidebar.markdown("---")

dataset_choice = st.sidebar.selectbox(
    "Select Cricket Format:",
    ["🏏 IPL", "🌍 ODI", "⚡ T20"],
    help="Choose the cricket format dataset"
)

dataset_map = {
    "🏏 IPL": "ipl.csv",
    "🌍 ODI": "odi.csv",
    "⚡ T20": "t20.csv"
}

csv_file = dataset_map[dataset_choice]

# Train the model
with st.spinner("🤖 Training model... (this may take a moment)"):
    model, scaler, X_test, y_test, y_pred = train_model(csv_file)

if model is not None:
    # Display metrics in sidebar (hidden - model info only when needed)
    st.sidebar.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Predict Score", "📋 Examples", "📈 Model Info"])
    
    with tab1:
        st.markdown("### Enter Match State")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Current Match**")
            runs = st.number_input("Current Runs", min_value=0, max_value=250, value=100)
            wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=2)
        
        with col2:
            st.markdown("**Overs**")
            overs = st.number_input("Overs Bowled", min_value=0.0, max_value=20.0, value=12.0)
            st.write("")
        
        with col3:
            st.markdown("**Batsmen**")
            striker = st.number_input("Striker Runs", min_value=0, max_value=150, value=50)
            non_striker = st.number_input("Non-striker Runs", min_value=0, max_value=150, value=40)
        
        st.markdown("---")
        
        if st.button("🎯 Predict Final Score", use_container_width=True, type="primary"):
            predicted = predict_score(model, scaler, runs, wickets, overs, striker, non_striker)
            std_error = np.std(y_pred - y_test)
            
            # Calculate upper bound and average
            upper_bound = predicted + std_error
            avg_score = (predicted + upper_bound) / 2
            
            # Display result - only average
            st.metric("Expected Final Score", f"{avg_score:.0f} runs")
            
            # Confidence info - show range
            st.success(
                f"🎯 **Expected Final Score: {avg_score:.0f} runs**\n\n"
                f"Based on range: {predicted:.0f} to {upper_bound:.0f} runs"
            )
    
    with tab2:
        st.markdown("### Example Predictions")
        
        examples = [
            (100, 0, 13, 50, 50, "Good start, no wickets"),
            (60, 3, 10, 30, 20, "Moderate start, some wickets"),
            (40, 5, 8, 15, 15, "Slow start, many wickets"),
            (150, 1, 15, 80, 60, "Great position"),
        ]
        
        cols = st.columns(2)
        std_error = np.std(y_pred - y_test)
        for idx, (r, w, o, s, ns, desc) in enumerate(examples):
            pred = predict_score(model, scaler, r, w, o, s, ns)
            upper = pred + std_error
            avg = (pred + upper) / 2
            with cols[idx % 2]:
                st.markdown(f"**{desc}**")
                st.write(f"📊 Match State:")
                st.write(f"  • Runs: `{r}` | Wickets: `{w}` | Overs: `{o}`")
                st.write(f"  • Striker: `{s}` | Non-striker: `{ns}`")
                st.success(f"➜ **Expected: {avg:.0f} runs**")
    
    with tab3:
        st.markdown("### 📊 Model Information")
        
        r2_score = model.score(scaler.transform(X_test), y_test) * 100
        acc_10 = custom_accuracy(y_test, y_pred, 10)
        acc_20 = custom_accuracy(y_test, y_pred, 20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Information**")
            st.markdown(f"""
            - **Test Samples:** {len(y_test):,}
            - **Algorithm:** Random Forest
            - **Features Used:** 5
              - Current runs
              - Wickets fallen
              - Overs bowled
              - Striker runs
              - Non-striker runs
            """)
        
        with col2:
            st.markdown("**Model Architecture**")
            st.code("""
RandomForestRegressor(
    n_estimators=100,
    max_features=None,
    random_state=0
)
            """, language="python")

else:
    st.error("❌ Could not load the model. Please check that all CSV files are present.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    <p>🏏 Cricket Score Predictor | Powered by Random Forest Machine Learning</p>
    <p style='font-size: 0.9em;'>Built with Streamlit | Datasets: IPL, ODI, T20</p>
    </div>
    """,
    unsafe_allow_html=True
)
