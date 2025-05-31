import librosa
import numpy as np
import os
import pickle

# Base path to your processed folder
base = r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed'

# Load audio paths
with open(f"{base}/audio_paths.pkl", "rb") as f:
    audio_paths = pickle.load(f)

# Load labels
labels = np.load(f"{base}/labels.npy")

mfccs = []
valid_labels = []

for idx, path in enumerate(audio_paths):
    try:
        y, sr = librosa.load(path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < 100:
            pad = 100 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode='constant')
        else:
            mfcc = mfcc[:, :100]
        mfccs.append(mfcc.T)
        valid_labels.append(labels[idx])  # Keep only matched label
    except Exception as e:
        print(f" Skipped {path} due to error: {e}")

# Convert and Save
mfcc_array = np.array(mfccs)
label_array = np.array(valid_labels)

np.save(f"{base}/audio_data.npy", mfcc_array)
np.save(f"{base}/labels_cleaned.npy", label_array)

print(" Extracted MFCCs:", mfcc_array.shape)
print(" Cleaned Labels:", label_array.shape)
