import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Add basedir to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import functions from the scripts module
from scripts.image_loader import load_and_process_images
from scripts.label_loader import load_labels
from scripts.resnet50v2_data_pipeline import process_images_and_labels

def build_model(input_shape, num_classes):
    """
    Build and compile the ResNet50V2 model.

    Parameters:
    - input_shape (tuple): Shape of the input images.
    - num_classes (int): Number of output classes.

    Returns:
    - Model: Compiled ResNet50V2 model.
    """
    print("Building the ResNet50V2 model...")
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Model built and compiled.")
    return model

if __name__ == "__main__":
    # Parameters
    IMAGE_SIZE = (224, 224)  # Image size for ResNet50V2
    NUM_CLASSES = 7  # Number of classes in your dataset (adjust as needed)
    BATCH_SIZE = 32
    EPOCHS = 10
    
    # Directories
    image_dir = 'datasets/raw/final_face_crops'
    labels_dir = 'datasets/raw/ground_truths'
    
    # Check if directories exist
    if not os.path.isdir(image_dir):
        print(f"Error: Image directory '{image_dir}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(labels_dir):
        print(f"Error: Labels directory '{labels_dir}' does not exist.")
        sys.exit(1)
    
    # Load and preprocess images and labels
    print("Loading and processing images...")
    images, image_filenames = load_and_process_images(image_dir)
    print(f"Loaded {len(images)} images.")
    
    print("Loading labels...")
    labels_dict = load_labels(labels_dir)
    print(f"Loaded labels dictionary: {labels_dict}")
    print(f"Type of loaded labels dictionary: {type(labels_dict)}")
    
    # Process images and labels
    print("Processing images and labels...")

    processed_images, processed_labels = process_images_and_labels(images, labels_dict, is_multi_label=True)
    
    # Convert to training and validation datasets
    print("Creating training and validation data generators...")
    train_datagen = ImageDataGenerator(validation_split=0.2)
    
    train_generator = train_datagen.flow(
        processed_images,
        processed_labels,
        batch_size=BATCH_SIZE,
        subset='training'
    )
    
    validation_generator = train_datagen.flow(
        processed_images,
        processed_labels,
        batch_size=BATCH_SIZE,
        subset='validation'
    )
    
    print(f"Training generator batches: {train_generator.samples // BATCH_SIZE}")
    print(f"Validation generator batches: {validation_generator.samples // BATCH_SIZE}")
    
    # Build and compile the model
    print("Building and compiling the model...")
    model = build_model(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES)
    
    # Check model summary
    print("Model summary:")
    model.summary()
    
    # Train the model
    print("Training the model...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS
    )
    
    # Save the trained model
    model_save_path = 'src/models/resnet50v2_trained_model.h5'
    print(f"Saving the model to {model_save_path}...")
    model.save(model_save_path)
    
    print("Training complete and model saved.")
