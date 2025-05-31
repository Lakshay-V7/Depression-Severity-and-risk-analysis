import scipy.io
import numpy as np
import os

def convert_mat_eeg_to_numpy(mat_dir, save_path):
   
    eeg_arrays = []

    print(f"\n Scanning EEG folder: {mat_dir}\n")

    # Iterate over sorted .mat files
    for idx, file in enumerate(sorted(os.listdir(mat_dir))):
        if file.endswith('.mat'):
            print(f" Processing file {idx + 1}: {file}")
            mat_path = os.path.join(mat_dir, file)
            mat = scipy.io.loadmat(mat_path)

            # Get the struct
            if 'EEG' in mat:
                eeg_struct = mat['EEG']

                # Debug info for the first file
                if idx == 0:
                    print(f"\n Available keys in {file}: {list(mat.keys())}")
                    print(" EEG struct type:", type(eeg_struct))
                    print(" EEG struct shape:", eeg_struct.shape)

                try:
                    # Extract EEG signal from struct
                    eeg_data = eeg_struct[0, 0]['data']  # ← adjust 'data' if needed
                    eeg_arrays.append(eeg_data)

                except Exception as e:
                    print(f"Could not extract EEG data from {file}: {e}")

            else:
                print(f" 'EEG' key not found in {file}")

    # Save all extracted EEG signals
    if eeg_arrays:
        eeg_np = np.array(eeg_arrays, dtype=object)  # object in case shapes vary
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, eeg_np)

        print(f"\n EEG data successfully saved to: {save_path}")
        print(f" Shape of saved array: {eeg_np.shape}")
    else:
        print("\n No EEG data was extracted. Please check struct contents.")


mat_dir = r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\eeg'
save_path = r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\eeg_data.npy'

convert_mat_eeg_to_numpy(mat_dir, save_path)
