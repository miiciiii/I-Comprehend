import onnx
from onnx_tf.backend import prepare
import os

# Define base paths for models and directories
BASE_MODEL_DIR = os.path.join('..', 'src', 'models')

# Define paths for ONNX and TensorFlow models
T5AG_MODEL_ONNX = os.path.join(BASE_MODEL_DIR, "t5ag_model.onnx")
T5AG_MODEL_TF = os.path.join(BASE_MODEL_DIR, "t5ag_model_tf")

T5QG_MODEL_ONNX = os.path.join(BASE_MODEL_DIR, "t5qg_model.onnx")
T5QG_MODEL_TF = os.path.join(BASE_MODEL_DIR, "t5qg_model_tf")

SENTENCE_TRANSFORMER_MODEL_ONNX = os.path.join(BASE_MODEL_DIR, "sentence_transformer_model.onnx")
SENTENCE_TRANSFORMER_MODEL_TF = os.path.join(BASE_MODEL_DIR, "sentence_transformer_model_tf")

RESNET50V2_MODEL_ONNX = os.path.join(BASE_MODEL_DIR, 'resnet50v2_model.onnx')
RESNET50V2_MODEL_TF = os.path.join(BASE_MODEL_DIR, 'resnet50v2_model_tf')

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

# # Convert ResNet50V2 Model to TensorFlow
# onnx_model = onnx.load(RESNET50V2_MODEL_ONNX)
# tf_rep = prepare(onnx_model)
# tf_rep.export_graph(RESNET50V2_MODEL_TF)

print("ONNX models successfully converted to TensorFlow format.")