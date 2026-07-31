
import streamlit as st
import pickle
import numpy as np

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Weather Forecast Predictor",
    page_icon="🌤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title-text {
        font-size: 42px;
        font-weight: 800;
        color: #1f2937;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle-text {
        font-size: 16px;
        color: #6b7280;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        font-weight: 600;
        font-size: 18px;
        padding: 12px 0px;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8, #1e40af);
        transform: scale(1.02);
    }
    .result-box {
        background-color: #ecfdf5;
        border-left: 6px solid #10b981;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        color: #065f46;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- LOAD MODEL -------------------
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    with open("logistic_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

FEATURE_NAMES = ["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"]

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/partly-cloudy-day--v1.png", width=80)
    st.header("About this App")
    st.write(
        "This application utilizes a trained **Random Forest Regressor** model "
        "to make predictions based on input weather conditions."
    )
    st.markdown("---")
    st.write("**Model Features:**")
    for f in FEATURE_NAMES:
        st.write(f"• {f}")
    st.markdown("---")
    st.caption("Developed with ❤️ using Streamlit")

# ------------------- HEADER -------------------
st.markdown('<p class="title-text">🌤️ Weather Forecast Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Set your weather parameters below to generate an instant prediction.</p>', unsafe_allow_html=True)

st.markdown("---")

# ------------------- INPUT FORM -------------------
st.subheader("📊 Input Weather Parameters")

col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("🌡️ Temperature (normalized)", 0.0, 1.0, 0.5, 0.01)
    wind_speed = st.slider("💨 Wind Speed (normalized)", 0.0, 1.0, 0.5, 0.01)
    pressure = st.slider("🧭 Pressure (normalized)", 0.0, 1.0, 0.5, 0.01)

with col2:
    humidity = st.slider("💧 Humidity (normalized)", 0.0, 1.0, 0.5, 0.01)
    cloud_cover = st.slider("☁️ Cloud Cover (normalized)", 0.0, 1.0, 0.5, 0.01)

st.markdown("---")

# ------------------- PREDICTION -------------------
if st.button("🔮 Predict Now"):
    input_data = np.array([[temperature, humidity, wind_speed, cloud_cover, pressure]])
    
    with st.spinner("Generating prediction from the model..."):
        prediction = model.predict(input_data)
    
    st.markdown(
        f'<div class="result-box">Predicted Value: {prediction[0]:.4f}</div>',
        unsafe_allow_html=True
    )
    
    st.balloons()

# ------------------- FOOTER -------------------
st.markdown("---")
st.caption("⚠️ Note: Predictions are based on normalized data. Apply inverse scaling to retrieve real-world values.")
