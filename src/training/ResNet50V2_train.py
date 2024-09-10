import sys
import os
import numpy as np
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scripts import load_and_process_images, load_labels, process_images_and_labels

def build_model(input_shape, num_classes):
    """
    Build and compile the ResNet50V2 model.

    Parameters:
    - input_shape (tuple): Shape of the input images.
    - num_classes (int): Number of output classes.

    Returns:
    - Model: Compiled ResNet50V2 model.
    """
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
    
    return model

if __name__ == "__main__":
    # Parameters
    IMAGE_SIZE = (224, 224)  # Image size for ResNet50V2
    NUM_CLASSES = 10  # Number of classes in your dataset (adjust as needed)
    BATCH_SIZE = 32
    EPOCHS = 10
    
    # Load and preprocess images and labels
    image_dir = 'datasets/raw/final_face_crops'
    labels_dir = 'datasets/raw/ground_truths'
    images, labels = load_and_process_images(image_dir)
    labels = load_labels(labels_dir)
    
    # Process images and labels
    processed_images, processed_labels = process_images_and_labels(images, labels, is_multi_label=True)
    
    # Convert to training and validation datasets
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
    
    # Build and compile the model
    model = build_model(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES)
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS
    )
    
    # Save the trained model
    model.save('resnet50v2_trained_model.h5')
