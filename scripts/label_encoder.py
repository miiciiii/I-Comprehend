import os
import json
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMAGE_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'raw', 'final_face_crops')
LABELS_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'processed', 'labels.json')

# Load labels from JSON
print("Loading labels from JSON...")
with open(LABELS_DATA_PATH, 'r') as f:
    labels = json.load(f)
print(f"Loaded {len(labels)} labels.")

# Parameters for chunking
chunk_size = 9000
image_data = []
labels_data = []
chunk_index = 0

# Create output directories if they don't exist
image_output_dir = os.path.join(BASE_DIR, 'datasets', 'processed', '2images')
labels_output_dir = os.path.join(BASE_DIR, 'datasets', 'processed', '2labels')
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(labels_output_dir, exist_ok=True)
print(f"Output directories created: {image_output_dir}, {labels_output_dir}")

# Create LabelEncoders for categorical labels
arousal_encoder = LabelEncoder()
dominance_encoder = LabelEncoder()

# Collect all unique classes for encoding
all_arousal = []
all_dominance = []

# First pass to collect unique classes
print("Collecting unique classes for arousal and dominance...")
for properties in labels.values():
    all_arousal.append(properties['arousal'].strip())
    all_dominance.append(properties['dominance'].strip())

# Fit the encoders
print("Fitting label encoders...")
arousal_encoder.fit(np.unique(all_arousal))
dominance_encoder.fit(np.unique(all_dominance))
print(f"Arousal classes: {arousal_encoder.classes_}")
print(f"Dominance classes: {dominance_encoder.classes_}")

# Iterate through tasks and images
print("Iterating through tasks and images...")
for task, properties in labels.items():
    task_dir = os.path.join(IMAGE_DATA_PATH, task)
    print(f"Processing task: {task}")

    if os.path.exists(task_dir):
        # Iterate through images in the task directory
        for image_name in os.listdir(task_dir):
            image_path = os.path.join(task_dir, image_name)

            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Loading image: {image_path}")
                try:
                    with Image.open(image_path) as img:
                        img = img.convert('RGB')
                        img_array = np.array(img) / 255.0
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue  # Skip to the next image

                image_data.append(img_array)

                # Encode the labels
                arousal_label = arousal_encoder.transform([properties['arousal'].strip()])[0]
                dominance_label = dominance_encoder.transform([properties['dominance'].strip()])[0]

                # Create a multi-class label array
                multi_class_label = [
                    arousal_label,
                    dominance_label,
                    properties['effort'],
                    properties['frustration'],
                    properties['mental_demand'],
                    properties['performance'],
                    properties['physical_demand']
                ]
                
                labels_data.append(multi_class_label)

                if len(image_data) >= chunk_size:
                    print(f"Saving chunk {chunk_index} with {len(image_data)} images and labels.")
                    np.save(os.path.join(image_output_dir, f'images_chunk_{chunk_index}.npy'), np.array(image_data))
                    np.save(os.path.join(labels_output_dir, f'labels_chunk_{chunk_index}.npy'), np.array(labels_data))
                    chunk_index += 1
                    image_data = []
                    labels_data = []

# Save any remaining data
if image_data:
    print(f"Saving remaining chunk {chunk_index} with {len(image_data)} images and labels.")
    np.save(os.path.join(image_output_dir, f'images_chunk_{chunk_index}.npy'), np.array(image_data))
    np.save(os.path.join(labels_output_dir, f'labels_chunk_{chunk_index}.npy'), np.array(labels_data))

# Save the encoders for future use
np.save(os.path.join(BASE_DIR, 'datasets', 'processed', 'arousal_encoder.npy'), arousal_encoder.classes_)
np.save(os.path.join(BASE_DIR, 'datasets', 'processed', 'dominance_encoder.npy'), dominance_encoder.classes_)

print("Labeling and saving completed.")
