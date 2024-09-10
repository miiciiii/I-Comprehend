import os
import pandas as pd
from sense2vec import Sense2Vec
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer

# Define paths for models and data
RANDOM_PASSAGE_PATH = os.path.join('..', 'datasets', 'processed', 'generated_qa.csv')
T5QG_MODEL_DIR = os.path.join('..', 'src', 'models', 't5_base_questiongeneration_model')
T5QG_TOKENIZER_DIR = os.path.join('..', 'src', 'models', 't5_base_questiongeneration_tokenizer')
T5AG_MODEL_DIR = os.path.join('..', 'src', 'models', 't5_base_answergeneration_model')
T5AG_TOKENIZER_DIR = os.path.join('..', 'src', 'models', 't5_base_answergeneration_tokenizer')
S2V_MODEL_PATH = os.path.join('..', 'src', 'models', 's2v_old')

# Load data and models
random_passage = pd.read_csv(RANDOM_PASSAGE_PATH)
t5ag_model = T5ForConditionalGeneration.from_pretrained(T5AG_MODEL_DIR)
t5ag_tokenizer = T5Tokenizer.from_pretrained(T5AG_TOKENIZER_DIR)
t5qg_model = T5ForConditionalGeneration.from_pretrained(T5QG_MODEL_DIR)
t5qg_tokenizer = T5Tokenizer.from_pretrained(T5QG_TOKENIZER_DIR)
s2v = Sense2Vec().from_disk(S2V_MODEL_PATH)
sentence_transformer_model = SentenceTransformer("sentence-transformers/LaBSE")

# Functions to get models and tokenizers
def get_answergeneration_model():
    return t5ag_model, t5ag_tokenizer

def get_questiongeneration_model():
    return t5qg_model, t5qg_tokenizer

def get_sense2vec_model():
    return s2v

def get_sentence_transformer_model():
    return sentence_transformer_model

def get_random_passage():
    return random_passage
