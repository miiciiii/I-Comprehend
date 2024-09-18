import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split

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
        image_data = np.load(IMAGE_DATA_PATH, mmap_mode='r')  # Memory mapping for large arrays
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

# Split data into training and validation sets
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels_encoded, test_size=0.2, random_state=42)

# Create a TensorFlow Dataset for efficient batch processing
def create_tf_dataset(X, y, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))  # Shuffle the dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # Prefetch to improve performance
    return dataset

# Create TensorFlow datasets for training and validation
batch_size = 32
train_dataset = create_tf_dataset(X_train, y_train, batch_size=batch_size)
val_dataset = create_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)

# Define the ResNet50V2 model
def create_resnet50v2_model(input_shape, num_classes):
    print("Creating ResNet50V2 model...")
    base_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the pre-trained layers
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    print("Model created.")
    return model, base_model

# Model configuration
input_shape = (256, 256, 3)
num_classes = 5  # Number of unique labels

# Create the model and base model
model, base_model = create_resnet50v2_model(input_shape, num_classes)

# Compile the model
print("Compiling the model...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

# Train the model using the TensorFlow dataset
print("Training the model...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Fine-tune the model after the initial training by unfreezing some layers
print("Fine-tuning the model...")
base_model.trainable = True  # Unfreeze the base model

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(val_dataset)
print(f"Validation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")

# Save the model
print(f"Saving the model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model training complete and saved.")
