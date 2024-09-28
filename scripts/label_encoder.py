import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib

# Constants
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths to your data
LABEL_DIR = os.path.join(BASE_DIR, 'datasets', 'processed', 'labels')
ENCODER_SAVE_PATH = os.path.join(BASE_DIR, 'outputs', 'label_encoder')
ENCODER_SAVE_PATH_AROUSAL = os.path.join(ENCODER_SAVE_PATH, 'label_encoder_arousal.pkl')
ENCODER_SAVE_PATH_DOMINANCE = os.path.join(ENCODER_SAVE_PATH, 'label_encoder_dominance.pkl')

# Load label files
label_files = [os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR) if f.endswith('.npy')]

# Collect all labels
all_arousal_labels = []
all_dominance_labels = []

for label_file in label_files:
    y = np.load(label_file, mmap_mode='r').astype('str')
    all_arousal_labels.extend(y[:, 0])   # Arousal labels
    all_dominance_labels.extend(y[:, 1]) # Dominance labels

# Convert to numpy arrays
all_arousal_labels = np.array(all_arousal_labels).reshape(-1, 1)
all_dominance_labels = np.array(all_dominance_labels).reshape(-1, 1)

# Fit OneHotEncoder on all labels
arousal_encoder = OneHotEncoder(sparse_output=False)
dominance_encoder = OneHotEncoder(sparse_output=False)

y_arousal_encoded = arousal_encoder.fit_transform(all_arousal_labels)
y_dominance_encoded = dominance_encoder.fit_transform(all_dominance_labels)

# Save the OneHotEncoders for later use
joblib.dump(arousal_encoder, ENCODER_SAVE_PATH_AROUSAL)
joblib.dump(dominance_encoder, ENCODER_SAVE_PATH_DOMINANCE)

print("Encoders have been saved.")
