import os 
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

LABEL_DIR = os.path.join(BASE_DIR, 'datasets', 'processed', 'labels', 'labels_chunk_0.npy')


print(f'Loading data from {LABEL_DIR}')

y = np.load(LABEL_DIR, mmap_mode='r')

print(f"Loaded and label data with shape {y.shape}.")

    # Check the shape and sample values of y
print(f"Shape of y: {y.shape}")
print("Sample values from y:")
print(y[:5])  # First 5 samples

