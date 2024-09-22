import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Define base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Define paths to data (use augmented dataset paths)
TRAIN_IMAGE_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'processed', 'images', 'images_chunk_1.npy')
TRAIN_LABELS_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'processed', 'labels', 'labels_chunk_1.npy')

MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'src', 'models', 'resnet50v2_model.keras')
PLOT_SAVE_DIR = os.path.join(BASE_DIR, 'outputs', 'plots', 'ResNet50V2')

# Ensure the plot directory exists
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# Load augmented training data
def load_augmented_data():
    print("Loading augmented training data...")
    image_data = np.load(TRAIN_IMAGE_DATA_PATH, mmap_mode='r')
    labels_data = np.load(TRAIN_LABELS_DATA_PATH, mmap_mode='r')
    
    print(f"Loaded {image_data.shape[0]} images.")
    print(f"Loaded {labels_data.shape[0]} labels.")
    return image_data, labels_data

# Convert labels to numeric data if necessary
def convert_labels_to_numeric(labels):
    # If labels are one-hot encoded, convert them to single numeric labels
    if labels.ndim == 2:
        print("Converting one-hot encoded labels to numeric...")
        labels = np.argmax(labels, axis=1)
    elif labels.dtype.kind in {'U', 'S', 'O'}:  # Check if labels are of string or object type
        print("Converting string labels to numeric...")
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        print(f"Unique labels: {le.classes_}")
    else:
        print("Labels are already numeric.")
    
    return labels

# Load augmented training data
X, y = load_augmented_data()

# Split the loaded data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% train, 20% validation

# Convert labels to numeric if they are not already
y_train = convert_labels_to_numeric(y_train)
y_val = convert_labels_to_numeric(y_val)

# Check shapes of loaded data
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Validation labels shape: {y_val.shape}")

# Convert labels to categorical if they are not already
num_classes = 7  # Number of unique labels, change as per your dataset

if len(y_train.shape) == 1 or y_train.shape[1] != num_classes:
    y_train = to_categorical(y_train, num_classes=num_classes)

if len(y_val.shape) == 1 or y_val.shape[1] != num_classes:
    y_val = to_categorical(y_val, num_classes=num_classes)

print(f"After encoding - Training labels shape: {y_train.shape}")
print(f"After encoding - Validation labels shape: {y_val.shape}")

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

    print("Model created.")
    return model

# Model configuration
input_shape = (256, 256, 3)  # Adjust if necessary

# Check if the model file exists and load it if available
best_model = None
best_val_accuracy = 0  # Initialize best validation accuracy

if os.path.exists(MODEL_SAVE_PATH):
    print(f"Loading model from {MODEL_SAVE_PATH}...")
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    print("Model loaded successfully.")
    _, best_val_accuracy = best_model.evaluate(X_val, y_val, verbose=0)  # Evaluate loaded model
    print(f"Current best validation accuracy: {best_val_accuracy:.4f}")
else:
    print("Model file not found. Creating a new model...")
    best_model = create_resnet50v2_model(input_shape, num_classes)

# Set up learning rate schedule and optimizer
initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate, 
    decay_steps=100000, 
    decay_rate=0.96, 
    staircase=True
)
optimizer = Adam(learning_rate=lr_schedule)

# Compile the model
print("Compiling the model...")
best_model.compile(optimizer=optimizer, 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])

# Print the model summary
print("Model summary:")
best_model.summary()

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss',  # Monitor validation loss
                               patience=3,          # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True)  # Restore model weights from the epoch with the best value of the monitored quantity

# Train the model
print("Training the model...")
history = best_model.fit(X_train, y_train,
                         validation_data=(X_val, y_val),
                         epochs=50,
                         batch_size=32,
                         verbose=1,
                         callbacks=[early_stopping])  # Include the early stopping callback

# Evaluate the new model
print("Evaluating the new model...")
new_loss, new_val_accuracy = best_model.evaluate(X_val, y_val)
print(f"New validation loss: {new_loss:.4f}")
print(f"New validation accuracy: {new_val_accuracy:.4f}")

# Save the new model only if it's better
if new_val_accuracy > best_val_accuracy:
    print(f"New model is better ({new_val_accuracy:.4f} > {best_val_accuracy:.4f}). Saving the new model...")
    best_model.save(MODEL_SAVE_PATH)
    print("Model saved.")
else:
    print(f"New model is not better ({new_val_accuracy:.4f} <= {best_val_accuracy:.4f}). Not saving the new model.")

# Plot the training history
def plot_training_history(history, save_dir):
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
    plot_path = os.path.join(save_dir, 'batch_2_training_history.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to {plot_path}")

# Call the function to plot the history and save the plot
plot_training_history(history, PLOT_SAVE_DIR)

print("Model training complete.")
