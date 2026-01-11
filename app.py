import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import io
from pydub import AudioSegment
import random
import pandas as pd
from tensorflow.keras.models import load_model
from streamlit_mic_recorder import mic_recorder
from huggingface_hub import hf_hub_download, list_repo_files

# UI Config
st.set_page_config(page_title="SER Neural Engine v2.5", layout="wide", page_icon="🧠")

# Session State to prevent Mic/Random interference
if 'active_audio' not in st.session_state:
    st.session_state.active_audio = None

@st.cache_resource
def load_production_assets():
    MODEL_REPO = "ShiroOnigami23/emotion-voice-engine"
    m_p = hf_hub_download(repo_id=MODEL_REPO, filename="emotion_brain.keras")
    s_p = hf_hub_download(repo_id=MODEL_REPO, filename="artifacts/scaler.joblib")
    e_p = hf_hub_download(repo_id=MODEL_REPO, filename="artifacts/label_encoder.joblib")
    return load_model(m_p), joblib.load(s_p), joblib.load(e_p)

model, scaler, lb = load_production_assets()

def process_signal(audio_source):
    audio_source.seek(0)
    try:
        # Step 1: Resample (16kHz kaiser_fast as per training)
        y, sr = librosa.load(audio_source, sr=16000, res_type='kaiser_fast')
    except:
        # Step 2: Rescue for Microphone formats
        audio_source.seek(0)
        audio = AudioSegment.from_file(audio_source).set_frame_rate(16000).set_channels(1)
        y = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        sr = 16000

    # Step 3: Normalize (Trim silence & Scale amplitude to 1.0)
    y, _ = librosa.effects.trim(y)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Step 4: Extract 40-Dim MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0).reshape(1, -1)
    
    # Step 5: Scale and Predict
    scaled = scaler.transform(features)
    return y, sr, mfccs, model.predict(scaled, verbose=0)[0]

# --- SIDEBAR ---
st.sidebar.title("🎛️ Engine Control Unit")
if st.sidebar.button("⚡ Run Random Neural Test"):
    DATA_REPO = "ShiroOnigami23/emotion-voice-dataset"
    all_files = list_repo_files(repo_id=DATA_REPO, repo_type="dataset")
    target = random.choice([f for f in all_files if f.endswith(".wav")])
    d_p = hf_hub_download(repo_id=DATA_REPO, filename=target, repo_type="dataset")
    with open(d_p, "rb") as f:
        st.session_state.active_audio = io.BytesIO(f.read())
    st.rerun()

st.sidebar.markdown("### 🎤 Live Bio-Telemetry")
mic_audio = mic_recorder(start_prompt="Record", stop_prompt="Stop", key='ser_mic')
if mic_audio:
    st.session_state.active_audio = io.BytesIO(mic_audio['bytes'])

# --- MAIN DASHBOARD ---
if st.session_state.active_audio:
    y, sr, mfccs, pred = process_signal(st.session_state.active_audio)
    label = lb.classes_[np.argmax(pred)].upper()
    
    st.metric("Detected Emotion", label, f"{np.max(pred)*100:.1f}% Confidence")
    st.audio(st.session_state.active_audio)
    
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(); librosa.display.waveshow(y, sr=sr, ax=ax1); st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(); librosa.display.specshow(mfccs, ax=ax2); st.pyplot(fig2)
else:
    st.info("Awaiting acoustic signal.")
    
