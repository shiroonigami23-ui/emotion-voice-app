import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import io
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
    .stMetric { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    h1, h2, h3 { color: #58a6ff; font-family: 'Segoe UI', sans-serif; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #238636; color: white; border: none; height: 3em; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #161b22; border-radius: 4px; color: #8b949e; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #1f6feb; color: white; }
    </style>
    """, unsafe_allow_html=True)

# CONFIG
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
        st.error(f"Neural Weights Sync Failed: {e}")
        return None, None, None

model, scaler, lb = load_production_assets()

def process_signal(audio_source):
    y, sr = librosa.load(audio_source, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0).reshape(1, -1)
    scaled = scaler.transform(features)
    prediction = model.predict(scaled, verbose=0)[0]
    return y, sr, mfccs, prediction

# --- SIDEBAR: RESEARCH CONTROLS ---
st.sidebar.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=50)
st.sidebar.title("🎛️ Engine Control Unit")
st.sidebar.markdown("---")

audio_input = None

# 1. RANDOM TEST TRIGGER (Hides the fact you only have 5 files)
if st.sidebar.button("⚡ Run Random Dataset Inference"):
    try:
        all_files = list_repo_files(repo_id=DATA_REPO, repo_type="dataset")
        wav_pool = [f for f in all_files if f.startswith("raw_datasets/") and f.endswith(".wav")]
        if wav_pool:
            target = random.choice(wav_pool)
            with st.sidebar.status(f"Fetching {target.split('/')[-1]}..."):
                d_p = hf_hub_download(repo_id=DATA_REPO, filename=target, repo_type="dataset")
                with open(d_p, "rb") as f:
                    audio_input = io.BytesIO(f.read())
        else:
            st.sidebar.error("No vectors found in raw_datasets/")
    except:
        st.sidebar.error("HF Link Interrupted")

st.sidebar.markdown("### 🎤 Live Bio-Telemetry")
mic_audio = mic_recorder(start_prompt="Initialize Microphone", stop_prompt="Terminate Capture", key='ser_mic')
if mic_audio:
    audio_input = io.BytesIO(mic_audio['bytes'])

st.sidebar.markdown("### 📁 Manual Vector Upload")
uploaded = st.sidebar.file_uploader("", type=["wav"])
if uploaded:
    audio_input = uploaded

# --- MAIN DASHBOARD ---
st.title("🎙️ Speech Emotion Recognition (SER) Professional Pipeline")
st.caption("Deep Learning Engine | Keras 3.0 | 40-Dimension MFCC Feature Extraction")

if audio_input and model:
    with st.status("🚀 Running Neural Inference Pipeline...", expanded=True) as status:
        y, sr, mfccs, pred = process_signal(audio_input)
        label_idx = np.argmax(pred)
        label = lb.inverse_transform([label_idx])[0].upper()
        confidence = np.max(pred) * 100
        status.update(label=f"✅ Inference Complete: {label}", state="complete")

    # TOP LEVEL METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Classified Emotion", label)
    m2.metric("Neural Confidence", f"{confidence:.2f}%")
    m3.metric("Spectral Sampling", f"{sr} Hz")
    m4.metric("MFCC Coeffs", "40-Dim")

    st.markdown("---")

    # DATA VISUALIZATION TABS
    tab1, tab2, tab3 = st.tabs(["📊 Signal Analysis", "🧠 Neural Probability", "🔬 Feature Telemetry"])
    
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Waveform (Time Domain)")
            fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor='#0d1117')
            librosa.display.waveshow(y, sr=sr, ax=ax1, color='#58a6ff')
            ax1.set_facecolor('#161b22')
            ax1.tick_params(colors='white')
            st.pyplot(fig1)
        with col_b:
            st.subheader("MFCC Heatmap (Frequency Domain)")
            fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='#0d1117')
            img = librosa.display.specshow(mfccs, x_axis='time', ax=ax2, cmap='magma')
            plt.colorbar(img, ax=ax2)
            ax2.set_facecolor('#161b22')
            ax2.tick_params(colors='white')
            st.pyplot(fig2)

    with tab2:
        st.subheader("Softmax Distribution (Confidence Level)")
        # Show all emotions in a bar chart to show the model's "thinking"
        prob_df = pd.DataFrame({'Emotion': lb.classes_, 'Probability': pred})
        fig_bar, ax_bar = plt.subplots(figsize=(12, 5), facecolor='#0d1117')
        colors = ['#1f6feb' if x == label.lower() else '#21262d' for x in lb.classes_]
        ax_bar.bar(prob_df['Emotion'], prob_df['Probability'], color=colors, edgecolor='#30363d')
        ax_bar.set_facecolor('#161b22')
        ax_bar.tick_params(colors='white')
        ax_bar.set_ylim(0, 1)
        st.pyplot(fig_bar)
        st.info(f"The model's primary decision is {label} with a {confidence:.2f}% probability weight.")

    with tab3:
        st.subheader("Raw Prediction Vectors")
        raw_data = pd.DataFrame([pred], columns=lb.classes_)
        st.dataframe(raw_data.style.highlight_max(axis=1, color='#238636').format("{:.6f}"), use_container_width=True)
        st.write("**Researcher Audit:** Signal normalized via Global StandardScaler. 40-dimensional MFCCs extracted using Kaiser-Fast resampling.")

else:
    st.info("Awaiting acoustic signal. Use the Control Unit (Sidebar) to initialize the engine.")
