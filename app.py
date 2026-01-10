
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import io
import pandas as pd
from tensorflow.keras.models import load_model
from streamlit_mic_recorder import mic_recorder

# --- THEME & ASSETS ---
st.set_page_config(page_title="SER Professional Engine", layout="wide")

# Custom CSS for Professional Scientific Branding
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    h1, h2, h3 { color: #f0f6fc; font-family: 'Helvetica Neue', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #161b22; border-radius: 5px; color: #8b949e; }
    .stTabs [aria-selected="true"] { background-color: #1f6feb; color: white; }
    </style>
    """, unsafe_allow_html=True)

# UPDATED PATHS for organized structure
MODEL_PATH = "/content/drive/MyDrive/Emotion_Project/models/emotion_brain.keras"
SCALER_PATH = "/content/drive/MyDrive/Emotion_Project/models/artifacts/scaler.joblib"
ENCODER_PATH = "/content/drive/MyDrive/Emotion_Project/models/artifacts/label_encoder.joblib"

@st.cache_resource
def load_assets():
    # Loading modern .keras format for better performance
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    lb = joblib.load(ENCODER_PATH)
    return model, scaler, lb

model, scaler, lb = load_assets()

# --- CORE LOGIC ---
def process_audio(audio_source):
    y, sr = librosa.load(audio_source, res_type='kaiser_fast')
    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfcc_feat.T, axis=0).reshape(1, -1)
    scaled = scaler.transform(features)
    pred = model.predict(scaled, verbose=0)[0]
    return y, sr, mfcc_feat, pred

# --- UI LAYOUT ---
st.title("🎙️ Professional Audio Emotion Pipeline")
st.caption("Developed by Gemini & Lead Researcher | Version 2.0 (Production-Ready)")

# Dual Input Section
st.sidebar.title("🎛️ Input Controls")
input_method = st.sidebar.radio("Select Input Source", ["🎤 Live Microphone", "📁 File Upload (.wav)"])

audio_input = None
if input_method == "🎤 Live Microphone":
    mic_audio = mic_recorder(start_prompt="Record Audio", stop_prompt="Stop Engine", key='recorder')
    if mic_audio:
        audio_input = io.BytesIO(mic_audio['bytes'])
else:
    uploaded_file = st.sidebar.file_uploader("Upload Spectral Data", type=["wav"])
    if uploaded_file:
        audio_input = uploaded_file

# --- MAIN DASHBOARD ---
if audio_input:
    y, sr, mfccs, pred = process_audio(audio_input)
    label = lb.inverse_transform([np.argmax(pred)])[0].upper()
    confidence = np.max(pred) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted State", label)
    m2.metric("Inference Confidence", f"{confidence:.2f}%")
    m3.metric("Spectral Sampling", f"{sr} Hz")
    m4.metric("MFCC Dimension", "40-Coef")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Spectral Heatmap", "🧠 Neural Distribution", "📋 Raw Telemetry"])
    
    with tab1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor='#0e1117')
        plt.subplots_adjust(hspace=0.5)
        librosa.display.waveshow(y, sr=sr, ax=ax1, color='#58a6ff')
        ax1.set_title("Waveform (Time-Domain)", color='white')
        ax1.set_facecolor('#161b22')
        ax1.tick_params(colors='white')

        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax2, cmap='magma')
        ax2.set_title("Spectral Density (MFCC)", color='white')
        ax2.set_facecolor('#161b22')
        ax2.tick_params(colors='white')
        fig.colorbar(img, ax=ax2, format="%+2.f dB")
        st.pyplot(fig)

    with tab2:
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5), facecolor='#0e1117')
        colors = ['#1f6feb' if (x == np.max(pred)) else '#21262d' for x in pred]
        ax_bar.bar(lb.classes_, pred, color=colors, edgecolor='#30363d')
        ax_bar.set_title("Softmax Prediction Confidence", color='white')
        ax_bar.set_facecolor('#161b22')
        ax_bar.tick_params(colors='white')
        ax_bar.set_ylim(0, 1)
        st.pyplot(fig_bar)

    with tab3:
        raw_df = pd.DataFrame([pred], columns=lb.classes_)
        st.dataframe(raw_df.style.highlight_max(axis=1, color='#238636').format("{:.6f}"))
        st.write("Professional Audit: Features normalized via StandardScaler. Inference via Keras 3 engine.")
else:
    st.warning("Awaiting signal input from sidebar.")

