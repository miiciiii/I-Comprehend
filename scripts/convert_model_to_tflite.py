import os
import torch
import transformers
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
from sentence_transformers import SentenceTransformer

# Define paths for models and directories
T5AG_MODEL_DIR = os.path.join('..', 'src', 'models', 't5_base_answergeneration_model')
T5AG_MODEL_ONNX = os.path.join('..', 'src', 'models', "t5ag_model.onnx")
T5AG_MODEL_TF = os.path.join('..', 'src', 'models', "t5ag_model_tf")
T5AG_MODEL_TFLITE = os.path.join('..', 'src', 'models', "t5ag_model.tflite")

T5QG_MODEL_DIR = os.path.join('..', 'src', 'models', 't5_base_questiongeneration_model')
T5QG_MODEL_ONNX = os.path.join('..', 'src', 'models', "t5qg_model.onnx")
T5QG_MODEL_TF = os.path.join('..', 'src', 'models', "t5qg_model_tf")
T5QG_MODEL_TFLITE = os.path.join('..', 'src', 'models', "t5qg_model.tflite")

SENTENCE_TRANSFORMER_MODEL_NAME = "sentence-transformers/LaBSE"
SENTENCE_TRANSFORMER_MODEL_ONNX = os.path.join('..', 'src', 'models', "sentence_transformer_model.onnx")
SENTENCE_TRANSFORMER_MODEL_TF = os.path.join('..', 'src', 'models', "sentence_transformer_model_tf")
SENTENCE_TRANSFORMER_MODEL_TFLITE = os.path.join('..', 'src', 'models', "sentence_transformer_model.tflite")

def convert_sentence_transformer_to_onnx(model_name, onnx_path):
    # Load SentenceTransformer model
    model = SentenceTransformer(model_name)
    
    # Create dummy input for ONNX export
    # Adjust the input based on the actual model requirements
    dummy_input = torch.tensor([[1, 2, 3, 4, 5]]).long()
    
    # Define the model's forward pass
    class TorchModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(TorchModelWrapper, self).__init__()
            self.model = model
        
        def forward(self, x):
            return self.model.encode(x, convert_to_tensor=True)
    
    wrapped_model = TorchModelWrapper(model)
    wrapped_model.eval()
    
    # Export the model to ONNX
    torch.onnx.export(
        wrapped_model, 
        dummy_input, 
        onnx_path, 
        verbose=True, 
        input_names=['input_ids'], 
        output_names=['output']
    )

def convert_pytorch_to_onnx(model_dir, onnx_path):
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    
    dummy_input = torch.tensor([[1, 2, 3]])  # Modify based on your input shape
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        verbose=True, 
        input_names=['input_ids'], 
        output_names=['output']
    )

def convert_onnx_to_tf(onnx_path, tf_path):
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_path)

def convert_tf_to_tflite(tf_path, tflite_path):
    # Load the TensorFlow model
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Ensure tflite_model is bytes
    if isinstance(tflite_model, bytes):
        # Write the model to a file
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
    else:
        raise TypeError("The converted model is not of type 'bytes'.")

# Convert SentenceTransformer model from PyTorch to ONNX
convert_sentence_transformer_to_onnx(SENTENCE_TRANSFORMER_MODEL_NAME, SENTENCE_TRANSFORMER_MODEL_ONNX)

# Convert T5 models from PyTorch to ONNX
convert_pytorch_to_onnx(T5AG_MODEL_DIR, T5AG_MODEL_ONNX)
convert_pytorch_to_onnx(T5QG_MODEL_DIR, T5QG_MODEL_ONNX)

# Convert ONNX models to TensorFlow
convert_onnx_to_tf(T5AG_MODEL_ONNX, T5AG_MODEL_TF)
convert_onnx_to_tf(T5QG_MODEL_ONNX, T5QG_MODEL_TF)

# Convert TensorFlow models to TensorFlow Lite
convert_tf_to_tflite(SENTENCE_TRANSFORMER_MODEL_TF, SENTENCE_TRANSFORMER_MODEL_TFLITE)
convert_tf_to_tflite(T5AG_MODEL_TF, T5AG_MODEL_TFLITE)
convert_tf_to_tflite(T5QG_MODEL_TF, T5QG_MODEL_TFLITE)

print("Conversion complete.")
