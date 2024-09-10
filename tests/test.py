# test_models.py (located in the tests directory)
import sys
import os

# Adjust the path to include the script directory where model_manager.py is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

# Now import the functions from model_manager
from model_manager import (
    get_answergeneration_model,
    get_questiongeneration_model,
    get_sense2vec_model,
    get_sentence_transformer_model,
    get_random_passage
)

# Example: Call the function to verify it works
t5ag_model, t5ag_tokenizer = get_answergeneration_model()
print("Answer generation model and tokenizer loaded successfully.")
