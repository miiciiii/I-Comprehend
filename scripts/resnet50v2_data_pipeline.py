from PIL import Image
import numpy as np
import os
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
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0
    return image_array

def multi_label_binarize(labels_list):
    """
    Convert multiple labels for each image to a binary format.

    Parameters:
    - labels_list (list of list of str): List where each element is a list of labels for a single image.

    Returns:
    - np.ndarray: Binarized labels.
    """
    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(labels_list)
    return binary_labels

def one_hot_encode_labels(labels_list):
    """
    Convert single labels to a one-hot encoded format.

    Parameters:
    - labels_list (list of str): List where each element is a single label for a single image.

    Returns:
    - np.ndarray: One-hot encoded labels.
    """
    lb = LabelBinarizer()
    one_hot_labels = lb.fit_transform(labels_list)
    return one_hot_labels

def process_images_and_labels(images, labels, is_multi_label=False, max_images=None):
    """
    Process images and labels for model training.

    Parameters:
    - images (list of PIL.Image.Image): List of images to process.
    - labels (list of list of str or list of str): Corresponding labels for images.
    - is_multi_label (bool): Indicates if labels are multi-label.
    - max_images (int): Maximum number of images to process.

    Returns:
    - np.ndarray: Processed images.
    - np.ndarray: Processed labels.
    """
    # Limit the number of images for testing
    if max_images is not None:
        images = images[:max_images]
        labels = labels[:max_images]
    
    print("Number of images:", len(images))  # Debugging statement
    print("Number of labels:", len(labels))  # Debugging statement
    
    # Preprocess images
    processed_images = np.array([preprocess_image(image) for image in images])
    
    # Process labels
    if is_multi_label:
        processed_labels = multi_label_binarize(labels)
    else:
        processed_labels = one_hot_encode_labels(labels)
    
    return processed_images, processed_labels

# Example usage
if __name__ == "__main__":
    # Parameters for testing
    # MAX_FOLDERS = 2
    # MAX_IMAGES_PER_FOLDER = 5
    # MAX_IMAGES = MAX_FOLDERS * MAX_IMAGES_PER_FOLDER

    # Load labels and images
    labels_dict = load_labels("datasets/raw/ground_truths")
    
    images, labels = load_and_process_images(
        image_dir='datasets/raw/final_face_crops'
        # max_folders=MAX_FOLDERS, 
        # max_images_per_folder=MAX_IMAGES_PER_FOLDER
    )
    
    processed_images, binary_labels = process_images_and_labels(images, labels, is_multi_label=True)