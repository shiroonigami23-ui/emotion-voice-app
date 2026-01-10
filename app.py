import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import io
import soundfile as sf
from pydub import AudioSegment
import random
import pandas as pd
from tensorflow.keras.models import load_model
from streamlit_mic_recorder import mic_recorder
from huggingface_hub import hf_hub_download, list_repo_files

# --- PRESTIGE UI CONFIG ---
st.set_page_config(page_title="SER Neural Engine v2.5", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .stMetric { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; }
    h1, h2, h3 { color: #58a6ff; font-family: 'Segoe UI', sans-serif; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #238636; color: white; border: none; height: 3em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'active_audio' not in st.session_state:
    st.session_state.active_audio = None
if 'last_source' not in st.session_state:
    st.session_state.last_source = None

MODEL_REPO = "ShiroOnigami23/emotion-voice-engine"
DATA_REPO = "ShiroOnigami23/emotion-voice-dataset"

@st.cache_resource
def load_production_assets():
    try:
        m_p = hf_hub_download(repo_id=MODEL_REPO, filename="emotion_brain.keras")
        s_p = hf_hub_download(repo_id=MODEL_REPO, filename="artifacts/scaler.joblib")
        e_p = hf_hub_download(repo_id=MODEL_REPO, filename="artifacts/label_encoder.joblib")
        return load_model(m_p), joblib.load(s_p), joblib.load(e_p)
    except Exception as e:
        st.error(f"⚠️ Engine Synchronization Failed: {e}")
        return None, None, None

model, scaler, lb = load_production_assets()

def process_signal(audio_source):
    """Strictly follows the preprocessing pipeline: Resample -> Trim -> Normalize -> MFCC"""
    audio_source.seek(0)
    
    try:
        # Step 1: Standard Load at 16kHz (Standard for your model)
        y, sr = librosa.load(audio_source, sr=16000, res_type='kaiser_fast')
    except Exception:
        # Fallback for raw browser/mic formats
        audio_source.seek(0)
        audio = AudioSegment.from_file(audio_source)
        audio = audio.set_frame_rate(16000).set_channels(1)
        sr = 16000
        y = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

    # Step 2: Trim Silence
    y, _ = librosa.effects.trim(y)
    
    # Step 3: Amplitude Normalization
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Step 4: Feature Extraction (40-Dim MFCC)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0).reshape(1, -1)
    
    # Step 5: Standardization & Inference
    scaled = scaler.transform(features)
    prediction = model.predict(scaled, verbose=0)[0]
    return y, sr, mfccs, prediction

# --- SIDEBAR: RESEARCH CONTROLS ---
st.sidebar.title("🎛️ Engine Control Unit")
st.sidebar.markdown("---")

# Clear Logic
if st.sidebar.button("🗑️ Clear Current Signal"):
    st.session_state.active_audio = None
    st.session_state.last_source = None
    st.rerun()

# 1. RANDOM TEST LOGIC
if st.sidebar.button("⚡ Random Neural Test"):
    try:
        all_files = list_repo_files(repo_id=DATA_REPO, repo_type="dataset")
        wav_pool = [f for f in all_files if f.startswith("samples/") and f.endswith(".wav")]
        if wav_pool:
            target = random.choice(wav_pool)
            with st.sidebar.status(f"Fetching: {target.split('/')[-1]}..."):
                d_p = hf_hub_download(repo_id=DATA_REPO, filename=target, repo_type="dataset")
                with open(d_p, "rb") as f:
                    st.session_state.active_audio = io.BytesIO(f.read())
                    st.session_state.last_source = "random"
    except Exception as e:
        st.sidebar.error("HF Connection Lost.")

st.sidebar.markdown("### 🎤 Live Bio-Telemetry")
# 2. MICROPHONE LOGIC
mic_data = mic_recorder(start_prompt="Initialize Microphone", stop_prompt="Terminate Capture", key='ser_mic')
if mic_data:
    # Use a specific key check to prevent automatic overwriting of random tests
    st.session_state.active_audio = io.BytesIO(mic_data['bytes'])
    st.session_state.last_source = "mic"

st.sidebar.markdown("### 📁 Manual Vector Upload")
# 3. UPLOAD LOGIC
uploaded = st.sidebar.file_uploader("Upload .wav signal", type=["wav"], key="manual_upload")
if uploaded:
    st.session_state.active_audio = uploaded
    st.session_state.last_source = "upload"

# --- MAIN DASHBOARD ---
st.title("🎙️ Speech Emotion Recognition Professional Pipeline")
st.caption("Deep Learning Engine | Keras 3.0 | 40-Dimension MFCC Feature Extraction")

# Retrieve from session state
audio_input = st.session_state.active_audio

if audio_input and model:
    try:
        with st.status("🚀 Running Neural Inference Pipeline...", expanded=True) as status:
            y, sr, mfccs, pred = process_signal(audio_input)
            label_idx = np.argmax(pred)
            label = lb.inverse_transform([label_idx])[0].upper()
            confidence = np.max(pred) * 100
            status.update(label=f"✅ Inference Complete: {label}", state="complete")

        # METRICS
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Classified Emotion", label)
        m2.metric("Neural Confidence", f"{confidence:.2f}%")
        m3.metric("Spectral Sampling", f"{sr} Hz")
        m4.metric("MFCC Coeffs", "40-Dim")
        
        # Audio Player (Resets seek to 0)
        audio_input.seek(0)
        st.audio(audio_input)
        st.markdown("---")

        # VISUALIZATION
        tab1, tab2, tab3 = st.tabs(["📊 Signal Analysis", "🧠 Neural Distribution", "🔬 Feature Telemetry"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor='#0d1117')
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='#58a6ff')
                ax1.set_facecolor('#161b22'); ax1.tick_params(colors='white')
                st.pyplot(fig1)
            with col_b:
                fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='#0d1117')
                img = librosa.display.specshow(mfccs, x_axis='time', ax=ax2, cmap='magma')
                ax2.set_facecolor('#161b22'); ax2.tick_params(colors='white')
                st.pyplot(fig2)

        with tab2:
            prob_df = pd.DataFrame({'Emotion': lb.classes_, 'Probability': pred})
            fig_bar, ax_bar = plt.subplots(figsize=(12, 5), facecolor='#0d1117')
            colors = ['#1f6feb' if (x == label.lower()) else '#21262d' for x in lb.classes_]
            ax_bar.bar(prob_df['Emotion'], prob_df['Probability'], color=colors)
            ax_bar.set_facecolor('#161b22'); ax_bar.tick_params(colors='white')
            st.pyplot(fig_bar)

        with tab3:
            st.write(f"Source: {st.session_state.last_source.upper()}")
            st.dataframe(pd.DataFrame([pred], columns=lb.classes_).style.highlight_max(axis=1, color='#238636'))

    except Exception as e:
        st.error(f"Signal Processing Error: {e}")
else:
    st.info("Awaiting acoustic signal. Use the Control Unit (Sidebar) to initialize the engine.")
            
