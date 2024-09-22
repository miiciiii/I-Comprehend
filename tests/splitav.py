import os
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMAGE_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'processed', 'image_data.npy')
LABELS_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'processed', 'image_labels_encoded.npy')

# Load original data
image_data = np.load(IMAGE_DATA_PATH, mmap_mode='r')
labels_data = np.load(LABELS_DATA_PATH, mmap_mode='r')

# Define chunk size
chunk_size = 10000  # Adjust this based on your needs

# Split image data into chunks
image_chunks = np.array_split(image_data, np.ceil(len(image_data) / chunk_size))
label_chunks = np.array_split(labels_data, np.ceil(len(labels_data) / chunk_size))

# Save each chunk as a separate .npy file
for i, (image_chunk, label_chunk) in enumerate(zip(image_chunks, label_chunks)):
    np.save(os.path.join(BASE_DIR, f'datasets/processed/image_data_chunk_{i}.npy'), image_chunk)
    np.save(os.path.join(BASE_DIR, f'datasets/processed/labels_data_chunk_{i}.npy'), label_chunk)

print(f"Saved {len(image_chunks)} image chunks and {len(label_chunks)} label chunks.")