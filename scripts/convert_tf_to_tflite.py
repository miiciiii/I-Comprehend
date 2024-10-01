import tensorflow as tf
import os

# Define base paths for models and directories
BASE_MODEL_DIR = os.path.join('..', 'src', 'models')
TFLITE_MODEL_DIR = os.path.join(BASE_MODEL_DIR, 'tensorflowmodels')

# Create the new directory for TFLite models if it doesn't exist
os.makedirs(TFLITE_MODEL_DIR, exist_ok=True)

# Define paths for TensorFlow and TFLite models
T5AG_MODEL_TF = os.path.join(BASE_MODEL_DIR, "t5ag_model_tf")
T5AG_MODEL_TFLITE = os.path.join(TFLITE_MODEL_DIR, "t5ag_model.tflite")

T5QG_MODEL_TF = os.path.join(BASE_MODEL_DIR, "t5qg_model_tf")
T5QG_MODEL_TFLITE = os.path.join(TFLITE_MODEL_DIR, "t5qg_model.tflite")

SENTENCE_TRANSFORMER_MODEL_TF = os.path.join(BASE_MODEL_DIR, "sentence_transformer_model_tf")
SENTENCE_TRANSFORMER_MODEL_TFLITE = os.path.join(TFLITE_MODEL_DIR, "sentence_transformer_model.tflite")

RESNET50V2_MODEL_TF = os.path.join(BASE_MODEL_DIR, 'resnet50v2_model_tf')
RESNET50V2_MODEL_TFLITE = os.path.join(TFLITE_MODEL_DIR, 'resnet50v2_model.tflite')

def convert_to_tflite(saved_model_dir, tflite_model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    
    # Ensure tflite_model is of type bytes
    if isinstance(tflite_model, bytes):
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Model successfully converted and saved to {tflite_model_path}")
    else:
        raise TypeError("The converted model is not of type bytes.")

# Convert models to TFLite
convert_to_tflite(T5AG_MODEL_TF, T5AG_MODEL_TFLITE)
convert_to_tflite(T5QG_MODEL_TF, T5QG_MODEL_TFLITE)
convert_to_tflite(SENTENCE_TRANSFORMER_MODEL_TF, SENTENCE_TRANSFORMER_MODEL_TFLITE)
# convert_to_tflite(RESNET50V2_MODEL_TF, RESNET50V2_MODEL_TFLITE)

print("TensorFlow models successfully converted to TFLite format.")