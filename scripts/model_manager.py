import os
import pandas as pd
from sense2vec import Sense2Vec
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer

RANDOM_PASSAGE_PATH = os.path.join('..', 'datasets', 'processed', 'generated_qa.csv')
T5QG_MODEL_DIR = os.path.join('..', 'src', 'models', 't5_base_questiongeneration_model')
T5QG_TOKENIZER_DIR = os.path.join('..', 'src', 'models', 't5_base_questiongeneration_tokenizer')
T5AG_MODEL_DIR = os.path.join('..', 'src', 'models', 't5_base_answergeneration_model')
T5AG_TOKENIZER_DIR = os.path.join('..', 'src', 'models', 't5_base_answergeneration_tokenizer')
S2V_MODEL_PATH = os.path.join('..', 'src', 'models', 's2v_old')


random_passage = pd.read_csv(RANDOM_PASSAGE_PATH)
t5ag_model = T5ForConditionalGeneration.from_pretrained(T5AG_MODEL_DIR)
t5ag_tokenizer = T5Tokenizer.from_pretrained(T5AG_TOKENIZER_DIR)
t5qg_model = T5ForConditionalGeneration.from_pretrained(T5QG_MODEL_DIR)
t5qg_tokenizer = T5Tokenizer.from_pretrained(T5QG_TOKENIZER_DIR)
s2v = Sense2Vec().from_disk(S2V_MODEL_PATH)
sentence_transformer_model = SentenceTransformer("sentence-transformers/LaBSE")

def get_answergeneration_model():
    return "hakdog"