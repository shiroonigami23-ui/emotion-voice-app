# 🎭 Professional Speech Emotion Recognition (SER) Engine

This project is a high-precision analytical system built for professional audio analysis. It utilizes Deep Learning (DNN) to analyze spectral vocal patterns and classify six core human emotions: **Happy, Sad, Angry, Fear, Disgust, and Neutral.**

---

## 📂 Project Navigation (Click to View)
Quickly access the technical reports and assets for this project:

- 🧠 **[Technical Architecture & Roadmap](./technical.md)**: Details on the DNN structure, framework choices, and future AI goals.
- 🎙️ **[Feature Engineering Report](./features.md)**: Deep dive into MFCC extraction and data preprocessing.
- 🏆 **[Project Summary](./summary.md)**: Executive overview of results, metrics, and final accuracy.
- ⚙️ **[App Requirements](./requirements.txt)**: List of Python libraries needed to run the app.

---

## 🎯 Achievement Highlights
- **Balanced Intelligence:** Trained on a perfectly balanced dataset (496 samples per class) ensuring zero bias across emotional categories.
- **High Accuracy:** Achieved **91.6% validation accuracy**, surpassing industry standard benchmarks for audio-only models.
- **Real-time Ready:** Low-latency inference engine providing spectral results in under 100ms.

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Audio Processing:** Librosa, Resampy (Kaiser-Fast Resampling)
- **Deep Learning:** TensorFlow 2.15, Keras 3
- **UI/UX:** Streamlit Professional Scientific Dashboard
- **Preprocessing:** Scikit-Learn, Joblib

---

## 📊 Visuals Gallery
The following telemetry results are available in the `visuals/` directory to verify model integrity:
- `training_metrics.png`: Learning curves (Accuracy/Loss) showing minimal variance.
- `confusion_matrix.png`: Heatmap demonstrating high class-separation precision.
- `model_architecture.png`: Schematic of the Neural Network hidden layers and dropout strategy.

---

## 📖 How to Use
1. **Environment Setup:** Navigate to the project folder and run `pip install -r Documentation/requirements.txt`.
2. **Launch Engine:** Initialize the dashboard by running `streamlit run app.py`.
3. **Analyze:** Utilize the sidebar to toggle between **Live Microphone** capture or **Spectral File Upload (.wav)**.

---
*Developed for Professional Audit | 2026*
