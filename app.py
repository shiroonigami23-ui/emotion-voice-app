import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import io
import soundfile as sf 
import random
import pandas as pd
from tensorflow.keras.models import load_model
from streamlit_mic_recorder import mic_recorder
from huggingface_hub import hf_hub_download, list_repo_files

st.set_page_config(page_title="SER Neural Engine v2.5", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .stMetric { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; }
    h1, h2, h3 { color: #58a6ff; font-family: 'Segoe UI', sans-serif; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #238636; color: white; border: none; height: 3em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

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
        # Prevents total crash if HF is down or path is wrong
        st.error(f"⚠️ Engine Synchronization Failed: {e}")
        return None, None, None

model, scaler, lb = load_production_assets()

def process_signal(audio_source):
    # Reset stream position to the beginning to ensure it's readable
    audio_source.seek(0)
    
    # Read the data and samplerate from the BytesIO object
    data, samplerate = sf.read(audio_source)
    
    # If the audio is stereo (2 channels), convert to mono for Librosa
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    # Use the data directly in librosa features
    y = data.astype(np.float32)
    sr = samplerate
    
    # Continue with your existing feature extraction
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0).reshape(1, -1)
    scaled = scaler.transform(features)
    prediction = model.predict(scaled, verbose=0)[0]
    
    return y, sr, mfccs, prediction

# --- SIDEBAR: RESEARCH CONTROLS ---
st.sidebar.title("🎛️ Engine Control Unit")
st.sidebar.markdown("---")

audio_input = None


if st.sidebar.button("⚡ Run Random Neural Test"):
    try:
        all_files = list_repo_files(repo_id=DATA_REPO, repo_type="dataset")
    
        wav_pool = [f for f in all_files if f.startswith("samples/") and f.endswith(".wav")]
        if wav_pool:
            target = random.choice(wav_pool)
            with st.sidebar.status(f"Fetching vector: {target.split('/')[-1]}..."):
                d_p = hf_hub_download(repo_id=DATA_REPO, filename=target, repo_type="dataset")
                with open(d_p, "rb") as f:
                    audio_input = io.BytesIO(f.read())
        else:
            st.sidebar.warning("No .wav files found in /samples/ folder.")
    except Exception as e:
        st.sidebar.error("HF Connection Timeout. Signal Lost.")

st.sidebar.markdown("### 🎤 Live Bio-Telemetry")
mic_audio = mic_recorder(start_prompt="Initialize Microphone", stop_prompt="Terminate Capture", key='ser_mic')
if mic_audio:
    audio_input = io.BytesIO(mic_audio['bytes'])

st.sidebar.markdown("### 📁 Manual Vector Upload")
uploaded = st.sidebar.file_uploader("Upload .wav signal", type=["wav"])
if uploaded:
    audio_input = uploaded

# --- MAIN DASHBOARD ---
st.title("🎙️ Speech Emotion Recognition Professional Pipeline")
st.caption("Deep Learning Engine | Keras 3.0 | 40-Dimension MFCC Feature Extraction")

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

        st.markdown("---")

        # VISUALIZATION
        tab1, tab2, tab3 = st.tabs(["📊 Signal Analysis", "🧠 Neural Distribution", "🔬 Feature Telemetry"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor='#0d1117')
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='#58a6ff')
                ax1.set_facecolor('#161b22')
                ax1.tick_params(colors='white')
                st.pyplot(fig1)
            with col_b:
                fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='#0d1117')
                img = librosa.display.specshow(mfccs, x_axis='time', ax=ax2, cmap='magma')
                plt.colorbar(img, ax=ax2)
                ax2.set_facecolor('#161b22')
                ax2.tick_params(colors='white')
                st.pyplot(fig2)

        with tab2:
            st.subheader("Softmax Distribution (Model Brain Decision)")
            prob_df = pd.DataFrame({'Emotion': lb.classes_, 'Probability': pred})
            fig_bar, ax_bar = plt.subplots(figsize=(12, 5), facecolor='#0d1117')
            colors = ['#1f6feb' if (x == label.lower()) else '#21262d' for x in lb.classes_]
            ax_bar.bar(prob_df['Emotion'], prob_df['Probability'], color=colors, edgecolor='#30363d')
            ax_bar.set_facecolor('#161b22')
            ax_bar.tick_params(colors='white')
            ax_bar.set_ylim(0, 1)
            st.pyplot(fig_bar)

        with tab3:
            st.subheader("Raw Prediction Vectors")
            raw_data = pd.DataFrame([pred], columns=lb.classes_)
            st.dataframe(raw_data.style.highlight_max(axis=1, color='#238636').format("{:.6f}"), use_container_width=True)
            st.write("**Researcher Audit:** Signal normalized via StandardScaler. Inference via Keras-TensorFlow.")

    except Exception as e:
        st.error(f"Signal Processing Error: {e}")
else:
    st.info("Awaiting acoustic signal. Use the Control Unit (Sidebar) to initialize the engine.")
