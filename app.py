# ==========================================
# MULTIMODAL STRESS DETECTION DASHBOARD
# ==========================================

import streamlit as st
import joblib
import librosa
import numpy as np
import re
import nltk
import plotly.graph_objects as go
import tempfile
import warnings
from streamlit_mic_recorder import mic_recorder

warnings.filterwarnings("ignore")

nltk.download("stopwords")
from nltk.corpus import stopwords

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Multimodal Stress Detection",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# GRADIENT UI STYLE
# ---------------------------------------------------

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Animated Gradient Background */

.stApp{
    background: linear-gradient(135deg,#667eea,#764ba2,#6dd5ed);
    background-size: 400% 400%;
    animation: gradient 12s ease infinite;
}

@keyframes gradient {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#FFC107,#FFF8E1);
}

/* Buttons */

.stButton>button{
    background: linear-gradient(45deg,#ff4e50,#f9d423);
    border-radius:25px;
    border:none;
    color:white;
    padding:10px 25px;
    font-weight:600;
}

.stButton>button:hover{
    transform:scale(1.05);
    background: linear-gradient(45deg,#24c6dc,#514a9d);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

text_model = joblib.load("text_stress_model.pkl")
audio_model = joblib.load("audio_stress_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("audio_scaler.pkl")

stop_words = set(stopwords.words("english"))

# ---------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------

def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# ---------------------------------------------------
# AUDIO FEATURE EXTRACTION
# ---------------------------------------------------

def extract_audio_features(path):

    y, sr = librosa.load(path, duration=5)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

    spectral = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

    tonnetz = np.mean(
        librosa.feature.tonnetz(
            y=librosa.effects.harmonic(y),
            sr=sr
        ).T,
        axis=0
    )

    features = np.hstack([mfcc, chroma, spectral, tonnetz])

    return features

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🧠 Stress Detection AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "📝 Text Stress Analysis",
        "🎤 Voice Stress Analysis",
        "🧠 Multimodal Fusion",
        "📈 Model Insights",
        "ℹ️ About Project"
    ]
)

# ---------------------------------------------------
# DASHBOARD
# ---------------------------------------------------

if page == "📊 Dashboard":

    st.title("AI Multimodal Stress Detection System")

    col1, col2, col3 = st.columns(3)

    col1.metric("📝 Text Model Accuracy", "84%")
    col2.metric("🎤 Audio Model Accuracy", "96.6%")
    col3.metric("🧠 Multimodal Accuracy", "93.3%")

    st.divider()

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=50,
        title={'text':"Average Stress Level"},
        gauge={
            'axis':{'range':[0,100]},
            'bar':{'color':"#ff4e50"},
            'steps':[
                {'range':[0,40],'color':'#2ecc71'},
                {'range':[40,70],'color':'#f1c40f'},
                {'range':[70,100],'color':'#e74c3c'}
            ]
        }
    ))

    st.plotly_chart(fig,use_container_width=True)

# ---------------------------------------------------
# TEXT ANALYSIS
# ---------------------------------------------------

elif page == "📝 Text Stress Analysis":

    st.title("📝 Text Stress Detection")

    text_input = st.text_area(
        "Enter text",
        placeholder="Type how you feel today..."
    )

    if st.button("Analyze Text"):

        clean = clean_text(text_input)

        vec = vectorizer.transform([clean])

        prob = text_model.predict_proba(vec)[0][1]

        stress = round(prob * 100,2)

        st.success(f"Stress Level: {stress}%")

# ---------------------------------------------------
# VOICE ANALYSIS
# ---------------------------------------------------

elif page == "🎤 Voice Stress Analysis":

    st.title("🎤 Voice Stress Detection")

    st.subheader("🎙 Record Voice")

    audio = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        key="voice_recorder"
    )

    if audio:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio['bytes'])
            path = tmp.name

        features = extract_audio_features(path)
        features = scaler.transform([features])

        prob = audio_model.predict_proba(features)[0][1]

        stress = round(prob * 100,2)

        st.success(f"Recorded Voice Stress Level: {stress}%")

    st.divider()

    st.subheader("📂 Upload Audio")

    uploaded = st.file_uploader("Upload WAV file", type=["wav"])

    if uploaded:

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded.read())
            path = tmp.name

        features = extract_audio_features(path)
        features = scaler.transform([features])

        prob = audio_model.predict_proba(features)[0][1]

        stress = round(prob * 100,2)

        st.success(f"Uploaded Audio Stress Level: {stress}%")

# ---------------------------------------------------
# MULTIMODAL FUSION
# ---------------------------------------------------

elif page == "🧠 Multimodal Fusion":

    st.title("🧠 Multimodal Stress Detection")

    text_input = st.text_area(
        "Enter text",
        placeholder="Type how you feel..."
    )

    st.subheader("🎙 Record Voice")

    audio = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        key="multi_recorder"
    )

    if st.button("Analyze Multimodal Stress"):

        if text_input and audio:

            clean = clean_text(text_input)
            vec = vectorizer.transform([clean])
            text_prob = text_model.predict_proba(vec)[0][1]

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(audio['bytes'])
                path = tmp.name

            features = extract_audio_features(path)
            features = scaler.transform([features])

            audio_prob = audio_model.predict_proba(features)[0][1]

            fusion_prob = 0.5 * text_prob + 0.5 * audio_prob

            stress = round(fusion_prob * 100,2)

            st.success(f"Multimodal Stress Level: {stress}%")

# ---------------------------------------------------
# MODEL INSIGHTS
# ---------------------------------------------------

elif page == "📈 Model Insights":

    st.title("📈 Model Insights")

    st.markdown("""
Text Model → TF-IDF + Stacking Ensemble  
Audio Model → MFCC + Spectral Features + SVM  

Multimodal Fusion → 50% Text + 50% Audio  

Text Accuracy: **84%**  
Audio Accuracy: **96.6%**  
Multimodal Accuracy: **93.3%**
""")

# ---------------------------------------------------
# ABOUT
# ---------------------------------------------------

elif page == "ℹ️ About Project":

    st.title("About Project")

    st.markdown("""
Multimodal Stress Detection System

Uses:

• Text sentiment analysis  
• Voice emotion analysis  

Built using:

Python  
Scikit-learn  
Librosa  
Streamlit
""")

st.markdown("---")

st.markdown(
"""
<center>
🧠 Multimodal Stress Detection System  
Final Year AI / ML Project
</center>
""",
unsafe_allow_html=True
)