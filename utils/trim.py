import numpy as np

# Load existing data
audio = np.load(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\audio_data.npy')
eeg = np.load(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\eeg_data_aligned.npy')
labels = np.load(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\labels.npy')

# Trim to match EEG size
audio_trimmed = audio[:396]
labels_trimmed = labels[:396]

# Confirm shapes
print(" Final Audio shape:", audio_trimmed.shape)
print(" Final EEG shape:", eeg.shape)
print(" Final Labels shape:", labels_trimmed.shape)

# (Optional) Save cleaned versions
np.save(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\audio_data_cleaned.npy', audio_trimmed)
np.save(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\labels_cleaned.npy', labels_trimmed)
