import os
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from tensorflow.keras.preprocessing import image

# Constants
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'src', 'models', 'resnet50v2_model.keras')
ENCODER_SAVE_PATH = os.path.join(BASE_DIR, 'outputs', 'label_encoder')
ENCODER_SAVE_PATH_AROUSAL = os.path.join(ENCODER_SAVE_PATH, 'label_encoder_arousal.pkl')
ENCODER_SAVE_PATH_DOMINANCE = os.path.join(ENCODER_SAVE_PATH, 'label_encoder_dominance.pkl')
PREDICTION_OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'predictions')

# Ensure output directory exists
os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

# Load the model
if os.path.exists(MODEL_SAVE_PATH):
    print(f"Loading model from {MODEL_SAVE_PATH}...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_SAVE_PATH}")

# Load OneHotEncoders
if os.path.exists(ENCODER_SAVE_PATH_AROUSAL):
    print(f"Loading OneHotEncoder for Arousal from {ENCODER_SAVE_PATH_AROUSAL}...")
    arousal_encoder = joblib.load(ENCODER_SAVE_PATH_AROUSAL)
else:
    raise FileNotFoundError(f"Arousal encoder file not found at {ENCODER_SAVE_PATH_AROUSAL}")

if os.path.exists(ENCODER_SAVE_PATH_DOMINANCE):
    print(f"Loading OneHotEncoder for Dominance from {ENCODER_SAVE_PATH_DOMINANCE}...")
    dominance_encoder = joblib.load(ENCODER_SAVE_PATH_DOMINANCE)
else:
    raise FileNotFoundError(f"Dominance encoder file not found at {ENCODER_SAVE_PATH_DOMINANCE}")

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    # Load image
    img = image.load_img(image_path, target_size=(256, 256))  # Resize to the expected input size
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image (if your model was trained on normalized data)
    return img_array

# Select the image for testing
image_file = r'D:\03PersonalFiles\Thesis\I-Comprehend\tests\example.png'  # Update this path as necessary
print(f"Loading image data from {image_file}...")

# Preprocess the image
X_test = load_and_preprocess_image(image_file)

# Make predictions
predictions = model.predict(X_test)

# Extract the prediction outputs
arousal_pred = predictions['arousal_output']
dominance_pred = predictions['dominance_output']
continuous_pred = predictions['continuous_output']

# Get the class index with the highest probability for arousal and dominance
arousal_class_index = np.argmax(arousal_pred, axis=1)
dominance_class_index = np.argmax(dominance_pred, axis=1)

# Inverse transform to get the string labels back
arousal_label = arousal_encoder.inverse_transform(np.eye(arousal_encoder.categories_[0].shape[0])[arousal_class_index].reshape(1, -1))
dominance_label = dominance_encoder.inverse_transform(np.eye(dominance_encoder.categories_[0].shape[0])[dominance_class_index].reshape(1, -1))

# Define the class names for the continuous output
class_names = ['effort', 'frustration', 'mental_demand', 'performance', 'physical_demand']

# Format and print the output
print("Prediction Results:")
print(f"Arousal: {arousal_label[0][0]}")  # Arousal label
print(f"Dominance: {dominance_label[0][0]}")  # Dominance label

# Map continuous output to corresponding class names
for i, name in enumerate(class_names):
    print(f"{name.capitalize()}: {continuous_pred[0][i]}")

print("Single image prediction completed.")
