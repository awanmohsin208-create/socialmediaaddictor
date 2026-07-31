
import streamlit as st
import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Social Media Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.main{
    background:#F5F7FB;
}

.hero{
    background:linear-gradient(135deg,#2563EB,#7C3AED);
    padding:35px;
    border-radius:20px;
    color:white;
    box-shadow:0px 8px 25px rgba(0,0,0,.18);
}

.hero h1{
    font-size:42px;
    font-weight:800;
}

.hero p{
    font-size:17px;
    color:#F3F4F6;
}

.feature-card{
    background:white;
    border-radius:15px;
    padding:18px;
    box-shadow:0px 5px 15px rgba(0,0,0,.08);
    text-align:center;
    transition:0.3s;
}

.feature-card:hover{
    transform:translateY(-5px);
}

.stButton>button{
    width:100%;
    background:linear-gradient(90deg,#2563EB,#7C3AED);
    color:white;
    font-size:18px;
    font-weight:bold;
    border:none;
    border-radius:12px;
    padding:14px;
}

.stButton>button:hover{
    background:linear-gradient(90deg,#1D4ED8,#6D28D9);
    color:white;
}

.result-box{
    background:#ECFDF5;
    padding:20px;
    border-radius:15px;
    border-left:8px solid #10B981;
    font-size:22px;
    font-weight:bold;
}

.sidebar-title{
    font-size:24px;
    font-weight:bold;
    color:#2563EB;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/label_encoders.pkl")
selected_features = joblib.load("models/selected_features.pkl")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.markdown("<div class='sidebar-title'>📱 Social Media Addiction</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🎯 Application Features")

    st.markdown("""
✅ AI-Powered Prediction

✅ Logistic Regression Model

✅ Smart Feature Inputs

✅ Real-Time Analysis

✅ Probability Estimation

✅ Responsive Dashboard

✅ Clean & Modern UI

✅ Instant Prediction
""")

    st.markdown("---")

    st.info(
        "This AI application predicts the likelihood of Social Media Addiction "
        "using a trained Machine Learning Logistic Regression model."
    )

# -----------------------------
# HERO SECTION
# -----------------------------
st.markdown("""
<div class="hero">

<h1>📱 Social Media Addiction Predictor</h1>

<p>
An AI-powered Machine Learning application that predicts the possibility of
<strong>Social Media Addiction</strong> using a trained
<strong>Logistic Regression Model</strong>.
<br><br>

Enter the required information below and click
<b>Predict</b> to receive an instant prediction along with confidence scores.
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------
# FEATURE CARDS
# -----------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="feature-card">
    <h2>🤖</h2>
    <h4>AI Prediction</h4>
    <p>Machine Learning Powered</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
    <h2>⚡</h2>
    <h4>Fast Results</h4>
    <p>Instant Prediction</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
    <h2>📊</h2>
    <h4>Probability</h4>
    <p>Confidence Score</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="feature-card">
    <h2>📱</h2>
    <h4>User Friendly</h4>
    <p>Professional Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.markdown("## 📝 Enter User Information")

# -----------------------------
# INPUT FORM
# -----------------------------
user_input = {}

col1, col2 = st.columns(2)

i = 0
for feature in selected_features:

    current_col = col1 if i % 2 == 0 else col2

    with current_col:

        if feature in encoders:

            options = list(encoders[feature].classes_)

            choice = st.selectbox(
                feature.replace("_"," ").title(),
                options
            )

            user_input[feature] = encoders[feature].transform([choice])[0]

        else:

            value = st.number_input(
                feature.replace("_"," ").title(),
                value=0.0
            )

            user_input[feature] = value

    i += 1

st.write("")

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚀 Predict Social Media Addiction"):

    input_df = pd.DataFrame([user_input])[selected_features]

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.markdown(
        f"""
        <div class="result-box">

        ✅ Prediction Result : <br><br>

        📌 <span style="color:#059669;font-size:28px;">
        {prediction}
        </span>

        </div>
        """,
        unsafe_allow_html=True
    )

    if hasattr(model, "predict_proba"):

        st.write("")

        st.subheader("📊 Prediction Confidence")

        probability = model.predict_proba(input_scaled)[0]

        proba_df = pd.DataFrame({
            "Class": model.classes_,
            "Probability": probability
        })

        st.dataframe(
            proba_df.style.format({"Probability":"{:.2%}"}),
            use_container_width=True
        )

        st.bar_chart(
            proba_df.set_index("Class")
        )

st.write("")

st.markdown("---")

st.caption(
    "© 2026 Social Media Addiction Predictor | "
    "Machine Learning • Logistic Regression • Streamlit Dashboard"
)
