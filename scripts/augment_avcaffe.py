import numpy as np
import tensorflow as tf
import os
import gc
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

# Define base directory and paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMAGE_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'processed', 'image_data_chunk_files')
LABELS_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'processed', 'labels_data_chunk_files')

# Debug: Check if directories exist
if not os.path.exists(IMAGE_DATA_PATH):
    print(f"Error: Image data path does not exist: {IMAGE_DATA_PATH}")
if not os.path.exists(LABELS_DATA_PATH):
    print(f"Error: Labels data path does not exist: {LABELS_DATA_PATH}")

# Initialize the ImageDataGenerator with your desired augmentations
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Desired number of augmented samples per chunk
desired_number_of_augmented_samples = 2000  # Adjust as needed
batch_size = 32  # Lowered batch size

# Process each of the first 7 chunks
for i in range(8):
    print(f"Processing chunk {i}...")

    # Load the current chunk
    image_chunk_path = os.path.join(IMAGE_DATA_PATH, f'image_data_chunk_{i}.npy')
    label_chunk_path = os.path.join(LABELS_DATA_PATH, f'labels_data_chunk_{i}.npy')

    # Check if the chunk files exist
    if not os.path.exists(image_chunk_path):
        print(f"Error: Image chunk file does not exist: {image_chunk_path}")
        continue
    if not os.path.exists(label_chunk_path):
        print(f"Error: Label chunk file does not exist: {label_chunk_path}")
        continue

    # Load the image and label data
    image_data_chunk = np.load(image_chunk_path).astype(np.float16)
    label_data_chunk = np.load(label_chunk_path).astype(np.float16)

    # Debug: Verify the shape and type of the loaded data
    print(f"Loaded image chunk {i} with shape: {image_data_chunk.shape} and dtype: {image_data_chunk.dtype}")
    print(f"Loaded label chunk {i} with shape: {label_data_chunk.shape} and dtype: {label_data_chunk.dtype}")

    # Fit the generator to the current chunk
    try:
        datagen.fit(image_data_chunk)
    except Exception as e:
        print(f"Error fitting ImageDataGenerator on chunk {i}: {e}")
        continue

    # Reset the lists for augmented images and labels
    augmented_images = []
    augmented_labels = []

    # Generate augmented images
    total_augmented_samples = 0
    for x_batch, y_batch in datagen.flow(image_data_chunk, label_data_chunk.astype(np.float16), batch_size=batch_size):
        augmented_images.append(x_batch.astype(np.float16))  # Convert to float16
        augmented_labels.append(y_batch.astype(np.float16))    # Convert to float16
        total_augmented_samples += len(x_batch)

        # Break after reaching the desired number of augmented samples
        if total_augmented_samples >= desired_number_of_augmented_samples:
            print(f"Reached desired augmented samples for chunk {i}")
            break

    # Convert to numpy arrays
    aug_images = np.concatenate(augmented_images, axis=0).astype(np.float16)  # Ensure float16
    aug_labels = np.concatenate(augmented_labels, axis=0).astype(np.float16)  # Ensure float16

    # Combine original and augmented images and labels
    combined_images = np.concatenate((image_data_chunk, aug_images), axis=0)
    combined_labels = np.concatenate((label_data_chunk, aug_labels), axis=0)

    # Debug: Check shapes before saving
    print(f"Saving combined data for chunk {i}: images shape {combined_images.shape}, labels shape {combined_labels.shape}")

    # Save the current combined data
    np.save(os.path.join(IMAGE_DATA_PATH, f'combined_training_images_chunk_{i}.npy'), combined_images)
    np.save(os.path.join(LABELS_DATA_PATH, f'combined_training_labels_chunk_{i}.npy'), combined_labels)

    # Clear variables to free up memory
    del image_data_chunk, label_data_chunk, augmented_images, augmented_labels, aug_images, aug_labels
    gc.collect()

# Final save for any remaining augmented images
print("Combined training data saved successfully.")

# Load the last two chunks for validation
try:
    validation_images = np.concatenate(
        [np.load(os.path.join(IMAGE_DATA_PATH, f'image_data_chunk_{i}.npy')).astype(np.float16) for i in range(8, 9)],
        axis=0
    )
    validation_labels = np.concatenate(
        [np.load(os.path.join(LABELS_DATA_PATH, f'labels_data_chunk_{i}.npy')).astype(np.float16) for i in range(8, 9)],
        axis=0
    )

    # Debug: Check validation data shapes
    print(f"Validation images shape: {validation_images.shape}")
    print(f"Validation labels shape: {validation_labels.shape}")

    # Save the validation data
    np.save(os.path.join(IMAGE_DATA_PATH, 'validation_images.npy'), validation_images)
    np.save(os.path.join(LABELS_DATA_PATH, 'validation_labels.npy'), validation_labels)
except Exception as e:
    print(f"Error loading or saving validation data: {e}")

print("Validation data saved successfully.")
