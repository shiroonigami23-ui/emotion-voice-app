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

# CONFIGURATION
REPO_ID = "ShiroOnigami23/emotion-voice-engine"

@st.cache_resource
def load_assets():
    with st.spinner("📥 Synchronizing Neural Weights from Hugging Face..."):
        # Downloading artifacts from your specific HF paths
        try:
            model_path = hf_hub_download(repo_id=REPO_ID, filename="emotion_brain.keras")
            scaler_path = hf_hub_download(repo_id=REPO_ID, filename="artifacts/scaler.joblib")
            encoder_path = hf_hub_download(repo_id=REPO_ID, filename="artifacts/label_encoder.joblib")
            
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            lb = joblib.load(encoder_path)
            return model, scaler, lb
        except Exception as e:
            st.error(f"Engine Failure: Could not sync with Hugging Face. Error: {e}")
            return None, None, None

model, scaler, lb = load_assets()

# --- CORE LOGIC ---
def process_audio(audio_source):
    # Load and resample
    y, sr = librosa.load(audio_source, res_type='kaiser_fast')
    # Feature Extraction (40 MFCCs)
    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfcc_feat.T, axis=0).reshape(1, -1)
    # Scaling
    scaled = scaler.transform(features)
    # Inference
    pred = model.predict(scaled, verbose=0)[0]
    return y, sr, mfcc_feat, pred

# --- UI LAYOUT ---
st.title("🎙️ Professional Audio Emotion Pipeline")
st.caption("Developed by ShiroOnigami & AI Thought Partner | Version 2.5 (High-Fidelity)")

# Dual Input Section
st.sidebar.title("🎛️ Input Controls")
st.sidebar.info("The engine expects 44.1kHz or 22kHz Mono/Stereo .wav signals.")
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
if audio_input and model is not None:
    # Adding the "Neural Telemetry" Status bar to look professional
    with st.status("🚀 Initializing Neural Pipeline...", expanded=True) as status:
        st.write("Extracting MFCC Spectrograms (40-Coefficient Space)...")
        y, sr, mfccs, pred = process_audio(audio_input)
        
        st.write("Applying Feature Scaling & Normalization...")
        label = lb.inverse_transform([np.argmax(pred)])[0].upper()
        confidence = np.max(pred) * 100
        
        status.update(label=f"✅ Inference Complete: {label}", state="complete", expanded=False)

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted State", label)
    m2.metric("Confidence Score", f"{confidence:.2f}%")
    m3.metric("Sampling Rate", f"{sr} Hz")
    m4.metric("Engine Base", "Keras/TensorFlow")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Spectral Heatmap", "🧠 Neural Distribution", "📋 Raw Telemetry"])
    
    with tab1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor='#0e1117')
        plt.subplots_adjust(hspace=0.5)
        
        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=ax1, color='#58a6ff')
        ax1.set_title("Waveform (Temporal Domain)", color='white', loc='left')
        ax1.set_facecolor('#161b22')
        ax1.tick_params(colors='white')

        # MFCC
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax2, cmap='magma')
        ax2.set_title("Spectral Density (MFCC Coefficients)", color='white', loc='left')
        ax2.set_facecolor('#161b22')
        ax2.tick_params(colors='white')
        fig.colorbar(img, ax=ax2, format="%+2.f dB")
        st.pyplot(fig)

    with tab2:
        st.subheader("Softmax Probability Distribution")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5), facecolor='#0e1117')
        colors = ['#1f6feb' if (x == np.max(pred)) else '#21262d' for x in pred]
        ax_bar.bar(lb.classes_, pred, color=colors, edgecolor='#30363d')
        ax_bar.set_facecolor('#161b22')
        ax_bar.tick_params(colors='white')
        ax_bar.set_ylim(0, 1)
        st.pyplot(fig_bar)

    with tab3:
        raw_df = pd.DataFrame([pred], columns=lb.classes_)
        st.dataframe(raw_df.style.highlight_max(axis=1, color='#238636').format("{:.6f}"), use_container_width=True)
        st.info("Note: Prediction represents the final activation layer output after standard scaling.")
elif model is None:
    st.error("Engine Offline. Please check Hugging Face Repository connectivity.")
else:
    st.info("Awaiting acoustic signal for processing...")
