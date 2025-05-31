import os
import numpy as np
import pickle

# Define correct base path for your audio dataset
base_path = r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\audio'

#  Create a folder to save processed outputs
save_dir = r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed'
os.makedirs(save_dir, exist_ok=True)

# Label Mapping
label_map = {
    'normal': 0,
    'stage1': 1,
    'stage2': 2
}

# Collect file paths and labels
labels = []
file_paths = []

for root, dirs, files in os.walk(base_path):
    for file in sorted(files):
        if file.endswith('.wav'):
            full_path = os.path.join(root, file)
            file_paths.append(full_path)

            # Assign label based on folder name
            if 'normal' in root.lower():
                labels.append(label_map['normal'])
            elif 'stage1' in root.lower():
                labels.append(label_map['stage1'])
            elif 'stage2' in root.lower():
                labels.append(label_map['stage2'])

# Save labels
labels_array = np.array(labels)
np.save(os.path.join(save_dir, 'labels.npy'), labels_array)

#  Save ordered file paths
with open(os.path.join(save_dir, 'audio_paths.pkl'), 'wb') as f:
    pickle.dump(file_paths, f)

# Confirmation
print(f" Saved {len(labels)} labels to labels.npy")
print(f" Saved {len(file_paths)} audio paths to audio_paths.pkl")
