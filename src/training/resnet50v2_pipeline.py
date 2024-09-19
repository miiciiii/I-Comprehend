import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping  #type: ignore

# Define base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Define paths to data
IMAGE_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'processed', 'image_data.npy')
LABELS_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'processed', 'image_labels_encoded.npy')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'src', 'models', 'resnet50v2_model.h5')

# Load preprocessed data using memory-mapped arrays
def load_preprocessed_data():
    print("Loading preprocessed data...")

    # Load image data
    if os.path.exists(IMAGE_DATA_PATH):
        image_data = np.load(IMAGE_DATA_PATH, mmap_mode='r')
    else:
        raise FileNotFoundError(f"Image data file not found at {IMAGE_DATA_PATH}.")

    # Load labels data
    if os.path.exists(LABELS_DATA_PATH):
        image_labels_encoded = np.load(LABELS_DATA_PATH, mmap_mode='r')
    else:
        raise FileNotFoundError(f"Labels data file not found at {LABELS_DATA_PATH}.")

    print(f"Loaded {image_data.shape[0]} images.")
    print(f"Loaded {image_labels_encoded.shape[0]} labels.")
    return image_data, image_labels_encoded

# Load data
image_data, image_labels_encoded = load_preprocessed_data()

# Check shapes of loaded data
print(f"Image data shape: {image_data.shape}")
print(f"Labels shape: {image_labels_encoded.shape}")

# Split data into training and validation sets
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels_encoded, test_size=0.2, random_state=42)

# Check shapes of split data
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation labels shape: {y_val.shape}")

# Define the ResNet50V2 model
def create_resnet50v2_model(input_shape, num_classes):
    print("Creating ResNet50V2 model...")
    base_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)  #type: ignore
    base_model.trainable = False

    model = tf.keras.Sequential([ #type: ignore
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(), #type: ignore
        tf.keras.layers.Dense(num_classes, activation='softmax') #type: ignore
    ])

    print("Model created.")
    return model

# Model configuration
input_shape = (256, 256, 3)  # Adjust if necessary
num_classes = 5  # Number of unique labels

# Create the model
model = create_resnet50v2_model(input_shape, num_classes)

# Compile the model
print("Compiling the model...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
print("Model summary:")
model.summary()

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss',  # Monitor validation loss
                               patience=3,          # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True)  # Restore model weights from the epoch with the best value of the monitored quantity

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10,
                    batch_size=32,
                    verbose=1,
                    callbacks=[early_stopping])  # Include the early stopping callback

# Print training history
print("Training history:")
print(f"History keys: {list(history.history.keys())}")
print(f"Final training loss: {history.history['loss'][-1]}")
print(f"Final training accuracy: {history.history['accuracy'][-1]}")
print(f"Final validation loss: {history.history['val_loss'][-1]}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]}")

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")

# Save the model
print(f"Saving the model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model training complete and saved.")
