import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import io
import pandas as pd
import os
from tensorflow.keras.models import load_model
from streamlit_mic_recorder import mic_recorder
from huggingface_hub import hf_hub_download

# --- THEME & ASSETS ---
st.set_page_config(page_title="SER Professional Engine", layout="wide", page_icon="🎙️")

# Force CSS Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    h1, h2, h3 { color: #f0f6fc; font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

REPO_ID = "ShiroOnigami23/emotion-voice-engine"

@st.cache_resource
def load_assets():
    try:
        # Pulling from your HF repo
        model_path = hf_hub_download(repo_id=REPO_ID, filename="emotion_brain.keras")
        scaler_path = hf_hub_download(repo_id=REPO_ID, filename="artifacts/scaler.joblib")
        encoder_path = hf_hub_download(repo_id=REPO_ID, filename="artifacts/label_encoder.joblib")
        
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        lb = joblib.load(encoder_path)
        return model, scaler, lb
    except Exception as e:
        st.error(f"HF Sync Error: {e}")
        return None, None, None

model, scaler, lb = load_assets()

# --- CORE LOGIC ---
def process_audio(audio_source):
    y, sr = librosa.load(audio_source, res_type='kaiser_fast')
    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfcc_feat.T, axis=0).reshape(1, -1)
    scaled = scaler.transform(features)
    pred = model.predict(scaled, verbose=0)[0]
    return y, sr, mfcc_feat, pred

# --- UI ---
st.title("🎙️ Professional Audio Emotion Pipeline")

# SIDEBAR
st.sidebar.title("📁 Sample Gallery")
sample_files = ["happy_sample.wav", "angry_sample.wav", "fear_sample.wav"] 
selected_sample = st.sidebar.selectbox("Test with Dataset", ["None"] + sample_files)

audio_input = None

if selected_sample != "None":
    sample_path = hf_hub_download(repo_id=REPO_ID, filename=f"samples/{selected_sample}")
    with open(sample_path, "rb") as f:
        audio_input = io.BytesIO(f.read())
else:
    input_method = st.sidebar.radio("Manual Input", ["🎤 Mic", "📁 Upload"])
    if input_method == "🎤 Mic":
        mic_audio = mic_recorder(start_prompt="Record", stop_prompt="Stop", key='recorder')
        if mic_audio: audio_input = io.BytesIO(mic_audio['bytes'])
    else:
        uploaded_file = st.sidebar.file_uploader("WAV file", type=["wav"])
        if uploaded_file: audio_input = uploaded_file

# DASHBOARD
if audio_input and model:
    with st.spinner("Analyzing Spectral Data..."):
        y, sr, mfccs, pred = process_audio(audio_input)
        label = lb.inverse_transform([np.argmax(pred)])[0].upper()
        confidence = np.max(pred) * 100

    st.subheader(f"Result: {label} ({confidence:.1f}%)")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)
    with col2:
        fig2, ax2 = plt.subplots()
        librosa.display.specshow(mfccs, x_axis='time', ax=ax2)
        st.pyplot(fig2)
