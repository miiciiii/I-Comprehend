import os
import json
import numpy as np
from PIL import Image

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMAGE_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'raw', 'final_face_crops')
LABELS_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'processed', 'labels.json')

# Load labels from JSON
with open(LABELS_DATA_PATH, 'r') as f:
    labels = json.load(f)

# Parameters for chunking
chunk_size = 9000
image_data = []
labels_data = []
chunk_index = 0

# Create output directories if they don't exist
image_output_dir = os.path.join(BASE_DIR, 'datasets', 'processed', 'images')
labels_output_dir = os.path.join(BASE_DIR, 'datasets', 'processed', 'labels')
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(labels_output_dir, exist_ok=True)

# Iterate through tasks and images
for task, properties in labels.items():
    task_dir = os.path.join(IMAGE_DATA_PATH, task)

    if os.path.exists(task_dir):
        # Iterate through images in the task directory
        for image_name in os.listdir(task_dir):
            image_path = os.path.join(task_dir, image_name)

            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    with Image.open(image_path) as img:
                        img = img.convert('RGB')
                        img_array = np.array(img) / 255.0
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue  # Skip to the next image

                image_data.append(img_array)

                multi_class_label = [
                    properties['arousal'].strip(),
                    properties['dominance'].strip(),
                    properties['effort'],
                    properties['frustration'],
                    properties['mental_demand'],
                    properties['performance'],
                    properties['physical_demand']
                ]
                
                labels_data.append(multi_class_label)

                if len(image_data) >= chunk_size:
                    np.save(os.path.join(image_output_dir, f'images_chunk_{chunk_index}.npy'), np.array(image_data))
                    np.save(os.path.join(labels_output_dir, f'labels_chunk_{chunk_index}.npy'), np.array(labels_data))
                    chunk_index += 1
                    image_data = []
                    labels_data = []

if image_data:
    np.save(os.path.join(image_output_dir, f'images_chunk_{chunk_index}.npy'), np.array(image_data))
    np.save(os.path.join(labels_output_dir, f'labels_chunk_{chunk_index}.npy'), np.array(labels_data))

print("Labeling and saving completed.")
