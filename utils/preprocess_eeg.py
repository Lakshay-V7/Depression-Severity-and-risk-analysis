import os
import scipy.io
import numpy as np

#  Paths
eeg_dir = r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\eeg'
save_path = r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\eeg_data.npy'

# Settings
max_timesteps = 500
target_channels = 66

eeg_array = []
eeg_files = sorted([f for f in os.listdir(eeg_dir) if f.endswith('.mat')])

for fname in eeg_files:
    try:
        mat = scipy.io.loadmat(os.path.join(eeg_dir, fname))
        eeg_struct = mat['EEG'][0, 0]
        data = eeg_struct['data']  # shape: (channels, time)

        if data.ndim != 2:
            print(f"Skipped {fname}: invalid dimensions {data.shape}")
            continue

        if data.shape[0] != target_channels:
            print(f"Skipped {fname}: expected {target_channels} channels, got {data.shape[0]}")
            continue

        # Transpose to (time, channels)
        data = data.T

        # Pad or trim
        if data.shape[0] > max_timesteps:
            data = data[:max_timesteps]
        elif data.shape[0] < max_timesteps:
            pad_len = max_timesteps - data.shape[0]
            data = np.vstack((data, np.zeros((pad_len, data.shape[1]))))

        if data.shape == (500, 66):
            eeg_array.append(data)
        else:
            print(f" Skipped {fname}: shape after pad/trim is {data.shape}")
    except Exception as e:
        print(f" Error in {fname}: {e}")

#  Convert to NumPy array
eeg_np = np.stack(eeg_array, axis=0)
np.save(save_path, eeg_np)

print(f"\n Saved EEG data to: {save_path}")
print(f" Final EEG shape: {eeg_np.shape}")
