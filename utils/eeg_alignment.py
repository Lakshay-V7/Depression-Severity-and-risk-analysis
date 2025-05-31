import os
import numpy as np
import scipy.io
import pickle

# Paths
eeg_dir = r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\eeg'
audio_paths_pkl = 'data/processed/audio_paths.pkl'

# 1. Load audio paths
with open(audio_paths_pkl, 'rb') as f:
    audio_paths = pickle.load(f)  # list of 400 paths

print(f"Loaded {len(audio_paths)} audio paths")

# 2. Build mapping from subject to EEG data
subject_to_eeg = {}
for filename in os.listdir(eeg_dir):
    if filename.endswith(".mat"):
        subj_id = os.path.splitext(filename)[0].lower()  # e.g., sub01
        mat_data = scipy.io.loadmat(os.path.join(eeg_dir, filename))
        eeg = mat_data[list(mat_data.keys())[-1]]  # grabs last key (likely the data)
        
        # Make sure EEG is (500, 66) per sample
        if eeg.shape[0] != 500:
            eeg = eeg.T  # transpose if needed

        subject_to_eeg[subj_id] = eeg
        print(f"Loaded EEG for {subj_id}: {eeg.shape}")

# 3. Create aligned EEG array
aligned_eeg = []

for path in audio_paths:
    # Extract subject ID from audio path
    filename = os.path.basename(path).lower()
    subj_id = filename.split('_')[0]  # assumes name like sub01_clip1.wav

    if subj_id not in subject_to_eeg:
        raise ValueError(f"Subject ID {subj_id} not found in EEG data.")

    aligned_eeg.append(subject_to_eeg[subj_id])  # shape: (500, 66)

aligned_eeg = np.array(aligned_eeg)  # shape: (400, 500, 66)
np.save("C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\eeg_data_aligned.npy", aligned_eeg)
print(" EEG aligned and saved. Final shape:", aligned_eeg.shape)
