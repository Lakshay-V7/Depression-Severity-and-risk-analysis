# Depression Severity and Risk Factor Analysis Using Multimodal Data (EEG + Audio)

This project explores the analysis and prediction of depression severity by leveraging **multimodal data** — combining **EEG signals** and **audio features**. We utilize advanced **machine learning** models such as **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks to capture both spatial and temporal dependencies in the data.

---

## 🔍 Objective

To develop a reliable and interpretable system for analyzing depression severity and identifying associated risk factors using EEG and audio data. This system is intended to assist in early diagnosis and personalized treatment recommendations.

---

## Modalities Used

- **EEG (Electroencephalogram):**
  - Captures electrical activity of the brain.
  - Preprocessed with filters, feature extraction (band powers, entropy, etc.).
  
- **Audio:**
  - Features extracted include MFCCs, pitch, energy, spectral features.
  - Normalized and processed into time series.

---

## 🧠 Models Used

- **CNN (Convolutional Neural Network):**
  - Used for spatial feature extraction from EEG signals.

- **LSTM (Long Short-Term Memory):**
  - Used for temporal pattern recognition in both EEG and audio time-series data.

The models were trained and evaluated separately and in combination to form a **multimodal fusion model**.

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | *95.6* |


#### ✅ Confusion Matrix

![Confusion Matrix](![image 1](https://github.com/user-attachments/assets/b968a334-b369-42a9-bb84-c4648287c371)
)



## Dataset

Due to ethical constraints, the dataset cannot be publicly released. However, the models were trained on anonymized and preprocessed EEG + audio data obtained from KAGGLEat YY kHz.

---

## 📦 Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt

