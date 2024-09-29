import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import joblib
import gc

# Constants
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
IMAGE_DIR = os.path.join(BASE_DIR, 'datasets', 'processed', 'images')
LABEL_DIR = os.path.join(BASE_DIR, 'datasets', 'processed', 'labels')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'src', 'models', 'resnet50v2_model.keras')
PLOT_SAVE_DIR = os.path.join(BASE_DIR, 'outputs', 'plots', 'ResNet50V2_plots_finaltraining')
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments', 'ResNet50V2_plots_finaltraining', 'logs')
ENCODER_SAVE_PATH = os.path.join(BASE_DIR, 'outputs', 'label_encoder')
ENCODER_SAVE_PATH_AROUSAL = os.path.join(ENCODER_SAVE_PATH, 'label_encoder_arousal.pkl')
ENCODER_SAVE_PATH_DOMINANCE = os.path.join(ENCODER_SAVE_PATH, 'label_encoder_dominance.pkl')
EPOCHS = 50
BATCH_SIZE = 64
TEST_SIZE = 0.2

# Ensure directories exist
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# Load data paths
image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.npy')]
label_files = [os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR) if f.endswith('.npy')]

if not image_files or not label_files:
    raise ValueError("No .npy files found in the specified directories.")

# Function to create the ResNet50V2 model
def create_resnet50v2_model(input_shape, num_arousal_classes, num_dominance_classes):
    base_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Arousal output (categorical)
    arousal_output = tf.keras.layers.Dense(num_arousal_classes, activation='softmax', name='arousal_output')(x)
    
    # Dominance output (categorical)
    dominance_output = tf.keras.layers.Dense(num_dominance_classes, activation='softmax', name='dominance_output')(x)

    # Continuous outputs (for the 5 continuous features)
    continuous_output = tf.keras.layers.Dense(5, activation='linear', name='continuous_output')(x)

    model = tf.keras.Model(inputs=inputs, outputs={'arousal_output': arousal_output, 
                                                    'dominance_output': dominance_output,
                                                    'continuous_output': continuous_output})
    
    model.summary()

    return model

# Function to train the model for a batch
def train_model(model, X_train, y_train_arousal, y_train_dominance, y_train_continuous, 
                X_val, y_val_arousal, y_val_dominance, y_val_continuous, batch_index):
    
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_arousal_output_accuracy', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_arousal_output_accuracy', patience=3, restore_best_weights=True, mode='max')

    history = model.fit(
        X_train,
        {'arousal_output': y_train_arousal, 'dominance_output': y_train_dominance, 'continuous_output': y_train_continuous},
        validation_data=(X_val, {'arousal_output': y_val_arousal, 'dominance_output': y_val_dominance, 'continuous_output': y_val_continuous}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_csv_path = os.path.join(EXPERIMENTS_DIR, f'batch_{batch_index + 1}_history.csv')
    history_df.to_csv(history_csv_path, index=False)

    # Plot training history
    plt.figure(figsize=(12, 6))
    
    # Arousal accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(history.history['arousal_output_accuracy'], label='Train Arousal Accuracy')
    plt.plot(history.history['val_arousal_output_accuracy'], label='Validation Arousal Accuracy')
    plt.title('Arousal Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Arousal loss plot
    plt.subplot(2, 2, 2)
    plt.plot(history.history['arousal_output_loss'], label='Train Arousal Loss')
    plt.plot(history.history['val_arousal_output_loss'], label='Validation Arousal Loss')
    plt.title('Arousal Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Dominance accuracy plot
    plt.subplot(2, 2, 3)
    plt.plot(history.history['dominance_output_accuracy'], label='Train Dominance Accuracy')
    plt.plot(history.history['val_dominance_output_accuracy'], label='Validation Dominance Accuracy')
    plt.title('Dominance Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Dominance loss plot
    plt.subplot(2, 2, 4)
    plt.plot(history.history['dominance_output_loss'], label='Train Dominance Loss')
    plt.plot(history.history['val_dominance_output_loss'], label='Validation Dominance Loss')
    plt.title('Dominance Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plot_path = os.path.join(PLOT_SAVE_DIR, f'batch_{batch_index + 1}_training_history.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

# Model configuration
input_shape = (256, 256, 3)
num_arousal_classes = 5  # For categories like excited, calm, etc.
num_dominance_classes = 5  # For categories like dependent, independent, etc.

# Load or create the OneHotEncoders
if os.path.exists(ENCODER_SAVE_PATH_AROUSAL):
    print(f"Loading OneHotEncoder for Arousal from {ENCODER_SAVE_PATH_AROUSAL}...")
    arousal_encoder = joblib.load(ENCODER_SAVE_PATH_AROUSAL)
else:
    arousal_encoder = OneHotEncoder(sparse_output=False)

if os.path.exists(ENCODER_SAVE_PATH_DOMINANCE):
    print(f"Loading OneHotEncoder for Dominance from {ENCODER_SAVE_PATH_DOMINANCE}...")
    dominance_encoder = joblib.load(ENCODER_SAVE_PATH_DOMINANCE)
else:
    dominance_encoder = OneHotEncoder(sparse_output=False)


# Training loop: start from the first batch and exclude the last batch
for i in range(5, len(image_files) - 1):  # Start from i=0 (first batch) and exclude last batch
    print(f"Loading data for batch {i + 1}...")
    try:
        # Load the data
        X = np.load(image_files[i], mmap_mode='r').astype('float32')
        y = np.load(label_files[i], mmap_mode='r').astype('str')

        # Separate arousal, dominance, and continuous labels
        arousal_labels = y[:, 0]  # Arousal is in the first column
        dominance_labels = y[:, 1]  # Dominance is in the second column
        continuous_labels = y[:, 2:].astype(float)  # Continuous features start from the third column

        # One-hot encoding for arousal and dominance using the loaded (pre-fitted) encoders
        y_arousal_encoded = arousal_encoder.transform(arousal_labels.reshape(-1, 1))
        y_dominance_encoded = dominance_encoder.transform(dominance_labels.reshape(-1, 1))

        # Split the data into training and validation sets
        X_train, X_val, y_train_arousal, y_val_arousal, y_train_dominance, y_val_dominance, y_train_continuous, y_val_continuous = train_test_split(
            X, y_arousal_encoded, y_dominance_encoded, continuous_labels, test_size=TEST_SIZE, random_state=42)

        # Create the model for each batch
        if os.path.exists(MODEL_SAVE_PATH):
            print(f"Loading model from {MODEL_SAVE_PATH}...")
            best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        else:
            print(f"No existing model found at {MODEL_SAVE_PATH}, creating a new one.")
            best_model = create_resnet50v2_model(input_shape, num_arousal_classes, num_dominance_classes)

        # Compile the model
        initial_learning_rate = 0.001
        lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
        optimizer = Adam(learning_rate=lr_schedule)
        best_model.compile(optimizer=optimizer,
                           loss={'arousal_output': 'categorical_crossentropy', 
                                 'dominance_output': 'categorical_crossentropy',
                                 'continuous_output': 'mean_squared_error'},
                           metrics={'arousal_output': ['accuracy'], 
                                    'dominance_output': ['accuracy']})

        # Train the model
        train_model(best_model, X_train, y_train_arousal, y_train_dominance, y_train_continuous, 
                    X_val, y_val_arousal, y_val_dominance, y_val_continuous, i)

        # Free up memory
        del X, y, y_arousal_encoded, y_dominance_encoded
        gc.collect()

    except Exception as e:
        print(f"An error occurred while processing batch {i + 1}: {str(e)}")
        break
