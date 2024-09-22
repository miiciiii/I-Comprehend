import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd  # For saving history

# Define base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(f"BASE_DIR: {BASE_DIR}")

EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')
print(f"EXPERIMENTS_DIR: {EXPERIMENTS_DIR}")

# Define paths to data (use augmented dataset paths)
IMAGE_DIR = os.path.join(BASE_DIR, 'datasets', 'processed', 'images')
LABEL_DIR = os.path.join(BASE_DIR, 'datasets', 'processed', 'labels')
print(f"IMAGE_DIR: {IMAGE_DIR}")
print(f"LABEL_DIR: {LABEL_DIR}")

# Collect all .npy files
image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.npy')]
label_files = [os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR) if f.endswith('.npy')]
print(f"Found {len(image_files)} image files and {len(label_files)} label files.")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'src', 'models', 'resnet50v2_model.keras')
PLOT_SAVE_DIR = os.path.join(BASE_DIR, 'outputs', 'plots', 'ResNet50V2_plots')

# Ensure directories exist
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)  # Create experiments directory if not exists
print(f"PLOT_SAVE_DIR: {PLOT_SAVE_DIR}")

# Convert labels to numeric data if necessary
def convert_labels_to_numeric(labels):
    if labels.ndim == 2:
        labels = np.argmax(labels, axis=1)
    elif labels.dtype.kind in {'U', 'S', 'O'}:
        le = LabelEncoder()
        labels = le.fit_transform(labels)
    return labels

# Function to create the ResNet50V2 model
def create_resnet50v2_model(input_shape, num_classes):
    print("Creating ResNet50V2 model...")
    base_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Model configuration
input_shape = (256, 256, 3)
num_classes = 7  # Adjust based on your dataset
best_model = None

# Check if the model file exists and load it if available
if os.path.exists(MODEL_SAVE_PATH):
    print(f"Loading model from {MODEL_SAVE_PATH}...")
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
else:
    print(f"No existing model found at {MODEL_SAVE_PATH}, creating a new one.")
    best_model = create_resnet50v2_model(input_shape, num_classes)

# Set up learning rate schedule and optimizer
initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
optimizer = Adam(learning_rate=lr_schedule)

# Compile the model
best_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print(f"Model compiled with input shape {input_shape} and {num_classes} classes.")

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Define the ModelCheckpoint callback to save the best model
model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

# Initialize variables to track previous best metrics
previous_val_loss = float('inf')
previous_val_accuracy = 0

# Iterate through each batch
for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
    print(f"\nLoading data from {image_file} and {label_file} (batch {i + 1})...")

    X = np.load(image_file, mmap_mode='r')
    y = np.load(label_file, mmap_mode='r')
    print(f"Loaded image data with shape {X.shape}, and label data with shape {y.shape}.")

    y = convert_labels_to_numeric(y)
    y = to_categorical(y, num_classes=num_classes)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split data into {X_train.shape[0]} training and {X_val.shape[0]} validation samples.")

    print(f"Training the model with batch {i + 1}...")
    history = best_model.fit(X_train, y_train,
                             validation_data=(X_val, y_val),
                             epochs=50,
                             batch_size=32,
                             verbose=1,
                             callbacks=[early_stopping, model_checkpoint])

    # Evaluate the model
    new_loss, new_val_accuracy = best_model.evaluate(X_val, y_val)
    print(f"New validation loss: {new_loss:.4f}")
    print(f"New validation accuracy: {new_val_accuracy:.4f}")

    # Update previous metrics
    previous_val_loss = new_loss
    previous_val_accuracy = new_val_accuracy

    # Save only the model weights for the current batch
    weights_save_path = os.path.join(EXPERIMENTS_DIR, 'weights', f'batch_{i + 1}_weights.h5')
    best_model.save_weights(weights_save_path)
    print(f"Model weights saved to {weights_save_path}.")

    # Log training history to CSV
    history_df = pd.DataFrame(history.history)
    history_csv_path = os.path.join(EXPERIMENTS_DIR, 'logs', f'batch_{i + 1}_history.csv')
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to {history_csv_path}.")

    # Save the training history plot for the current batch
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    # Save the plot
    plot_path = os.path.join(PLOT_SAVE_DIR, f'batch_{i + 1}_training_history.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}.")

print("Model training complete.")
