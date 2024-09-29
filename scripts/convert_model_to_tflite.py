import os
import torch
import transformers
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
from sentence_transformers import SentenceTransformer

# Define base paths for models and directories
BASE_MODEL_DIR = os.path.join('..', 'src', 'models')
TFLITE_MODEL_DIR = os.path.join(BASE_MODEL_DIR, 'tensorflowmodels')

# Create the new directory for TFLite models if it doesn't exist
os.makedirs(TFLITE_MODEL_DIR, exist_ok=True)

# Define paths for models and directories
T5AG_MODEL_DIR = os.path.join(BASE_MODEL_DIR, 't5_base_answergeneration_model')
T5AG_MODEL_ONNX = os.path.join(BASE_MODEL_DIR, "t5ag_model.onnx")
T5AG_MODEL_TF = os.path.join(BASE_MODEL_DIR, "t5ag_model_tf")
T5AG_MODEL_TFLITE = os.path.join(TFLITE_MODEL_DIR, "t5ag_model.tflite")

T5QG_MODEL_DIR = os.path.join(BASE_MODEL_DIR, 't5_base_questiongeneration_model')
T5QG_MODEL_ONNX = os.path.join(BASE_MODEL_DIR, "t5qg_model.onnx")
T5QG_MODEL_TF = os.path.join(BASE_MODEL_DIR, "t5qg_model_tf")
T5QG_MODEL_TFLITE = os.path.join(TFLITE_MODEL_DIR, "t5qg_model.tflite")

SENTENCE_TRANSFORMER_MODEL_NAME = "sentence-transformers/LaBSE"
SENTENCE_TRANSFORMER_MODEL_ONNX = os.path.join(BASE_MODEL_DIR, "sentence_transformer_model.onnx")
SENTENCE_TRANSFORMER_MODEL_TF = os.path.join(BASE_MODEL_DIR, "sentence_transformer_model_tf")
SENTENCE_TRANSFORMER_MODEL_TFLITE = os.path.join(TFLITE_MODEL_DIR, "sentence_transformer_model.tflite")

RESNET50V2_MODEL_DIR = os.path.join(BASE_MODEL_DIR, 'resnet50v2_model.keras')
RESNET50V2_MODEL_ONNX = os.path.join(BASE_MODEL_DIR, 'resnet50v2_model.onnx')
RESNET50V2_MODEL_TF = os.path.join(BASE_MODEL_DIR, 'resnet50v2_model_tf')
RESNET50V2_MODEL_TFLITE = os.path.join(TFLITE_MODEL_DIR, 'resnet50v2_model.tflite')

# Step 1: Load the models
# Load T5 Answer Generation Model
t5ag_model = transformers.T5ForConditionalGeneration.from_pretrained(T5AG_MODEL_DIR)

# Load T5 Question Generation Model
t5qg_model = transformers.T5ForConditionalGeneration.from_pretrained(T5QG_MODEL_DIR)

# Load Sentence Transformer Model
sentence_transformer_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)

# Load ResNet50V2 Model
resnet50v2_model = tf.keras.models.load_model(RESNET50V2_MODEL_DIR)

# Step 2: Convert models to ONNX
# Convert T5 Answer Generation Model to ONNX
torch.onnx.export(t5ag_model, torch.randn(1, 512), T5AG_MODEL_ONNX)

# Convert T5 Question Generation Model to ONNX
torch.onnx.export(t5qg_model, torch.randn(1, 512), T5QG_MODEL_ONNX)

# Convert Sentence Transformer Model to ONNX
torch.onnx.export(sentence_transformer_model, torch.randn(1, 512), SENTENCE_TRANSFORMER_MODEL_ONNX)

# Convert ResNet50V2 Model to ONNX
onnx_model = onnx.load(RESNET50V2_MODEL_DIR)
onnx.save(onnx_model, RESNET50V2_MODEL_ONNX)

# Step 3: Convert ONNX to TensorFlow
# Convert T5 Answer Generation Model to TensorFlow
onnx_model = onnx.load(T5AG_MODEL_ONNX)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(T5AG_MODEL_TF)

# Convert T5 Question Generation Model to TensorFlow
onnx_model = onnx.load(T5QG_MODEL_ONNX)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(T5QG_MODEL_TF)

# Convert Sentence Transformer Model to TensorFlow
onnx_model = onnx.load(SENTENCE_TRANSFORMER_MODEL_ONNX)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(SENTENCE_TRANSFORMER_MODEL_TF)

# Convert ResNet50V2 Model to TensorFlow
onnx_model = onnx.load(RESNET50V2_MODEL_ONNX)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(RESNET50V2_MODEL_TF)

# Step 4: Convert TensorFlow to TFLite
def convert_to_tflite(saved_model_dir, tflite_model_path):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Ensure tflite_model is of type bytes
    if isinstance(tflite_model, bytes):
        # Write the model to a .tflite file
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Model successfully converted and saved to {tflite_model_path}")
    else:
        print("Error: The converted model is not of type bytes.")

# Convert models to TFLite
convert_to_tflite(T5AG_MODEL_TF, T5AG_MODEL_TFLITE)
convert_to_tflite(T5QG_MODEL_TF, T5QG_MODEL_TFLITE)
convert_to_tflite(SENTENCE_TRANSFORMER_MODEL_TF, SENTENCE_TRANSFORMER_MODEL_TFLITE)
convert_to_tflite(RESNET50V2_MODEL_TF, RESNET50V2_MODEL_TFLITE)