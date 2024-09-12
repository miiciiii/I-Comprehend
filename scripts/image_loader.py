from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from .label_loader import load_labels

# Default directory for images
DEFAULT_IMAGE_DIR = "datasets/raw/final_face_crops"

def load_images_and_labels(image_dir, labels_dict, max_folders=None, max_images_per_folder=None):
    """
    Load images and their labels from the specified directory.

    Parameters:
    - image_dir (str): Directory containing task folders with images.
    - labels_dict (dict): Dictionary mapping task names to labels.
    - max_folders (int): Maximum number of folders to process.
    - max_images_per_folder (int): Maximum number of images to process per folder.

    Returns:
    - images (list of PIL.Image.Image): List of PIL Image objects.
    - labels (list of dict): List of labels corresponding to the images.
    """
    print(f"Checking image directory: {image_dir}")
    
    if not os.path.isdir(image_dir):
        raise ValueError(f"The directory {image_dir} does not exist or is not a directory.")
    
    images = []
    labels = []
    task_folders = os.listdir(image_dir)
    
    if max_folders is not None:
        task_folders = task_folders[:max_folders]
    print(f"Found task folders: {task_folders}")

    for task_name in tqdm(task_folders, desc="Loading Images"):
        task_path = os.path.join(image_dir, task_name)
        print(f"Processing task folder: {task_path}")
        
        if os.path.isdir(task_path) and task_name in labels_dict:
            image_files = [f for f in os.listdir(task_path) if f.endswith('.jpg') or f.endswith('.png')]
            print(f"Found {len(image_files)} image files in folder '{task_name}'")

            if max_images_per_folder is not None:
                image_files = image_files[:max_images_per_folder]
            print(f"Processing {len(image_files)} images from folder '{task_name}'")

            for image_name in image_files:
                image_path = os.path.join(task_path, image_name)
                try:
                    print(f"Loading image: {image_path}")
                    image = Image.open(image_path).convert('RGB')
                    images.append(image)
                    # Retrieve all labels for the task
                    labels.append(labels_dict[task_name])
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    
    print(f"Total images loaded: {len(images)}")
    print(f"Total labels loaded: {len(labels)}")
    return images, labels

def load_and_process_images(image_dir=DEFAULT_IMAGE_DIR, max_folders=None, max_images_per_folder=None):
    """
    Load labels and images from the default or specified directory, and return them.

    Parameters:
    - image_dir (str): Directory containing task folders with images. Defaults to DEFAULT_IMAGE_DIR.
    - max_folders (int): Maximum number of folders to process.
    - max_images_per_folder (int): Maximum number of images to process per folder.

    Returns:
    - images (list of PIL.Image.Image): List of PIL Image objects.
    - labels (list of dict): List of labels corresponding to the images.
    """
    try:
        print(f"Loading labels from 'datasets/raw/ground_truths'")
        labels_dict = load_labels("datasets/raw/ground_truths")
        print(f"Labels loaded: {labels_dict.keys()}")
        print(f"Loading images from directory: {image_dir}")
        return load_images_and_labels(image_dir, labels_dict, max_folders=max_folders, max_images_per_folder=max_images_per_folder)
    except Exception as e:
        print(f"Error loading and processing images: {e}")
        return [], []
