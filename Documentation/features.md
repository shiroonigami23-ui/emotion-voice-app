# 🎙️ Audio Feature Engineering Report

## 1. Dataset Overview
This model was trained on a balanced combination of **RAVDESS, TESS, CREMA-D, and SAVEE** datasets.

- **Total Samples:** 2976
- **Features extracted per sample:** 40 (MFCCs)
- **Input Shape:** (2976, 40)

### 📊 Class Distribution (Balanced)
- **Angry:** 496 samples
- **Disgust:** 496 samples
- **Fear:** 496 samples
- **Happy:** 496 samples
- **Neutral:** 496 samples
- **Sad:** 496 samples

## 2. Feature Extraction Methodology: MFCCs
The primary features used for this model are **Mel-Frequency Cepstral Coefficients (MFCCs)**. 

### Why MFCC?
MFCCs represent the short-term power spectrum of a sound. In Speech Emotion Recognition (SER), they are the industry standard because they mimic how the human ear perceives frequency, making them highly effective at capturing:
- **Tone/Pitch:** Distinguishing between high-energy (Happy/Angry) and low-energy (Sad) emotions.
- **Timbre:** Capturing the unique "texture" of an emotional voice.



## 3. Preprocessing Pipeline
1.  **Resampling:** All audio files resampled to a consistent rate using `kaiser_fast`.
2.  **Normalization:** Silence was trimmed and audio amplitude was normalized.
3.  **Averaging:** MFCCs were calculated over time-frames and averaged to create a 1D feature vector of length 40.
4.  **Standardization:** Used `StandardScaler` to ensure all features have a mean of 0 and a variance of 1, preventing high-volume features from dominating the model.

## 4. Final Data Integrity
- **Missing Values:** 0
- **Duplicate Paths:** 0
- **Standardized Range:** ~[-3, 3]
