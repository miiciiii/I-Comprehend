import torch
import transformers
import os
# import tensorflow as tf
import onnx
from sentence_transformers import SentenceTransformer
# import tf2onnx

# Define base paths for models and directories
BASE_MODEL_DIR = os.path.join('..', 'src', 'models')

# Define paths for models
T5AG_MODEL_DIR = os.path.join(BASE_MODEL_DIR, 't5_base_answergeneration_model')
T5AG_MODEL_ONNX = os.path.join(BASE_MODEL_DIR, "t5ag_model.onnx")

T5QG_MODEL_DIR = os.path.join(BASE_MODEL_DIR, 't5_base_questiongeneration_model')
T5QG_MODEL_ONNX = os.path.join(BASE_MODEL_DIR, "t5qg_model.onnx")

SENTENCE_TRANSFORMER_MODEL_NAME = "sentence-transformers/LaBSE"
SENTENCE_TRANSFORMER_MODEL_ONNX = os.path.join(BASE_MODEL_DIR, "sentence_transformer_model.onnx")

RESNET50V2_MODEL_DIR = os.path.join(BASE_MODEL_DIR, 'resnet50v2_model.keras')
RESNET50V2_MODEL_ONNX = os.path.join(BASE_MODEL_DIR, 'resnet50v2_model.onnx')

# Convert T5 Answer Generation Model to ONNX
t5ag_model = transformers.T5ForConditionalGeneration.from_pretrained(T5AG_MODEL_DIR)
torch.onnx.export(t5ag_model, torch.randn(1, 512), T5AG_MODEL_ONNX)

# Convert T5 Question Generation Model to ONNX
t5qg_model = transformers.T5ForConditionalGeneration.from_pretrained(T5QG_MODEL_DIR)
torch.onnx.export(t5qg_model, torch.randn(1, 512), T5QG_MODEL_ONNX)

# Convert Sentence Transformer Model to ONNX
sentence_transformer_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
torch.onnx.export(sentence_transformer_model, torch.randn(1, 512), SENTENCE_TRANSFORMER_MODEL_ONNX)

# # Convert ResNet50V2 Model to ONNX using tf2onnx
# resnet50v2_model = tf.keras.models.load_model(RESNET50V2_MODEL_DIR)
# spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
# model_proto, _ = tf2onnx.convert.from_keras(resnet50v2_model, input_signature=spec, opset=13)
# onnx.save(model_proto, RESNET50V2_MODEL_ONNX)

print("Models successfully converted to ONNX format.")