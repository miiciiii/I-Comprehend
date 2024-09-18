import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import json
from tqdm import tqdm
from label_loader import load_labels

def load_labels_from_file(output_file="datasets/processed/labels.json"):
    """Load labels from a saved file."""
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_labels = json.load(f)
        return all_labels
    else:
        raise FileNotFoundError(f"No label file found at {output_file}")

def load_images_and_labels(images_path, labels):
    """Load images from a directory structure and associate them with their labels."""
    image_data = []
    image_labels = []
    tasks = os.listdir(images_path)
    
    for task in tqdm(tasks, desc="Processing Tasks"):
        task_path = os.path.join(images_path, task)
        if os.path.isdir(task_path):
            image_files = [f for f in os.listdir(task_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for image_file in tqdm(image_files, desc=f"Processing Images in {task}", leave=False):
                image_path = os.path.join(task_path, image_file)
                try:
                    # Load and preprocess image
                    with Image.open(image_path) as img:
                        img = img.convert('RGB')
                        img_array = np.array(img)
                        image_data.append(img_array)
                    
                    # Get the corresponding label
                    if task in labels:
                        metric = list(labels[task].keys())[0]  # Assuming the first metric
                        image_labels.append(labels[task][metric])
                    else:
                        print(f"No labels found for task: {task}")

                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    
    return np.array(image_data), np.array(image_labels)

def prepare_data_for_training():
    ground_truths_path = "D:/03PersonalFiles/Thesis/I-Comprehend/datasets/raw/ground_truths"
    images_path = "D:/03PersonalFiles/Thesis/I-Comprehend/datasets/raw/final_face_crops"
    labels_file = "D:/03PersonalFiles/Thesis/I-Comprehend/datasets/processed/labels.json"
    
    # Ensure labels are loaded and saved
    load_labels(ground_truths_path, labels_file)
    
    # Load labels
    labels = load_labels_from_file(labels_file)
    
    # Load images and corresponding labels
    image_data, image_labels = load_images_and_labels(images_path, labels)
    
    # Encode labels as integers
    label_encoder = LabelEncoder()
    image_labels_encoded = label_encoder.fit_transform(image_labels)
    
    print(f"Number of images loaded: {len(image_data)}")
    print(f"Number of labels: {len(image_labels)}")
    print(f"Unique labels: {len(label_encoder.classes_)}")
    print(f"Label classes: {label_encoder.classes_}")
    
    # Check some images
    if len(image_data) > 0:
        print(f"Sample image shape: {image_data[0].shape}")
        print(f"Sample image label: {image_labels_encoded[0]}") # type: ignore
    
    return image_data, image_labels_encoded, label_encoder

def loader():
    """Loader function to check if the code is working."""
    print("Starting data preparation and loading...")
    
    # Ensure the output directory exists
    output_dir = "D:/03PersonalFiles/Thesis/I-Comprehend/datasets/processed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_data, image_labels_encoded, label_encoder = prepare_data_for_training()
    
    # Save the preprocessed data
    np.save(os.path.join(output_dir, "image_data.npy"), image_data)
    np.save(os.path.join(output_dir, "image_labels_encoded.npy"), image_labels_encoded)
    
    print("Data preparation complete.")
    print("Preprocessed data and labels have been saved.")

if __name__ == "__main__":
    loader()
