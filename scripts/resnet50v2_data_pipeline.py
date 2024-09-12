from PIL import Image
import numpy as np
import os
import gc
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from tqdm import tqdm
from .image_loader import load_and_process_images
from .label_loader import load_labels

def preprocess_image(image, target_size=(224, 224)):
    """
    Resize and normalize an image.

    Parameters:
    - image (PIL.Image.Image): The input image.
    - target_size (tuple): The target size (width, height) for resizing.

    Returns:
    - np.ndarray: The preprocessed and normalized image as a NumPy array.
    """
    print(f"Preprocessing image of size {image.size}")
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float16)  # Use float16 to save memory
    image_array = image_array / 255.0
    print(f"Image array shape after preprocessing: {image_array.shape}")
    return image_array

def multi_label_binarize(labels_list):
    """
    Convert multiple labels for each image to a binary format.

    Parameters:
    - labels_list (list of list of str): List where each element is a list of labels for a single image.

    Returns:
    - np.ndarray: Binarized labels.
    """
    print(f"Multi-label binarizing with {len(labels_list)} entries.")
    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(labels_list)
    print(f"Binary labels shape: {binary_labels.shape}")
    return binary_labels

def one_hot_encode_labels(labels_list):
    """
    Convert single labels to a one-hot encoded format.

    Parameters:
    - labels_list (list of str): List where each element is a single label for a single image.

    Returns:
    - np.ndarray: One-hot encoded labels.
    """
    print(f"One-hot encoding {len(labels_list)} labels.")
    lb = LabelBinarizer()
    one_hot_labels = lb.fit_transform(labels_list)
    print(f"One-hot labels shape: {one_hot_labels.shape}")
    return one_hot_labels

def process_images_and_labels(images, labels, is_multi_label=False, max_images=None, batch_size=32):
    """
    Process images and labels for model training.

    Parameters:
    - images (list of PIL.Image.Image): List of images to process.
    - labels (list of list of str or list of str): Corresponding labels for images.
    - is_multi_label (bool): Indicates if labels are multi-label.
    - max_images (int): Maximum number of images to process.
    - batch_size (int): Number of images to process in each batch.

    Returns:
    - np.ndarray: Processed images.
    - np.ndarray: Processed labels.
    """
    # Limit the number of images for testing
    if max_images is not None:
        images = images[:max_images]
        labels = labels[:max_images]
    
    print(f"Processing {len(images)} images.")
    
    # Initialize lists to store processed images and labels
    processed_images = []
    processed_labels = []
    
    # Process images and labels in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # Process each image in the batch
        batch_processed_images = [preprocess_image(image) for image in batch_images]
        processed_images.extend(batch_processed_images)
        
        # Process labels
        if is_multi_label:
            batch_processed_labels = multi_label_binarize(batch_labels)
        else:
            batch_processed_labels = one_hot_encode_labels(batch_labels)
        
        processed_labels.extend(batch_processed_labels) # type: ignore
        
        # Free up memory
        del batch_processed_images
        del batch_labels
        gc.collect()
    
    # Convert lists to arrays
    processed_images = np.array(processed_images, dtype=np.float16)  # Use float16 for processed images
    processed_labels = np.array(processed_labels, dtype=np.float32)  # One-hot encoded labels as float32
    
    return processed_images, processed_labels

if __name__ == "__main__":
    # Parameters for testing
    # MAX_FOLDERS = 2
    # MAX_IMAGES_PER_FOLDER = 5
    # MAX_IMAGES = MAX_FOLDERS * MAX_IMAGES_PER_FOLDER

    # Load labels and images
    try:
        print("Loading labels from 'datasets/raw/ground_truths'")
        labels_dict = load_labels("datasets/raw/ground_truths")
        print(f"Labels loaded: {len(labels_dict)} tasks.")

        print("Loading images from 'datasets/raw/final_face_crops'")
        images, labels = load_and_process_images(
            image_dir='datasets/raw/final_face_crops'
            # max_folders=MAX_FOLDERS, 
            # max_images_per_folder=MAX_IMAGES_PER_FOLDER
        )
        print(f"Loaded {len(images)} images and {len(labels)} labels.")

        processed_images, binary_labels = process_images_and_labels(images, labels, is_multi_label=True)
        print(f"Processed images shape: {processed_images.shape}")
        print(f"Processed labels shape: {binary_labels.shape}")

    except Exception as e:
        print(f"Error in main execution: {e}")
