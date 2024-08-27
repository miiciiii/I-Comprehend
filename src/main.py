import os
import warnings
import random
import string

import numpy as np
import pandas as pd
import nltk
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from sense2vec import Sense2Vec
from textdistance import levenshtein
import pke
from nltk.corpus import stopwords, wordnet as wn

# Download NLTK data
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')

# Configure warnings
warnings.filterwarnings("ignore", message="This sequence already has </s>.")


# Define base directory as the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the base directory
RANDOM_PASSAGE_PATH = os.path.join(BASE_DIR, 'datasets', 'processed', 'generated_qa.csv')

T5QG_MODEL_DIR = os.path.join(BASE_DIR, 'models', 't5_base_questiongeneration_model')
T5QG_TOKENIZER_DIR = os.path.join(BASE_DIR, 'models', 't5_base_questiongeneration_tokenizer')
T5AG_MODEL_DIR = os.path.join(BASE_DIR, 'models', 't5_base_answergeneration_model')
T5AG_TOKENIZER_DIR = os.path.join(BASE_DIR, 'models', 't5_base_answergeneration_tokenizer')

S2V_MODEL_PATH = os.path.join(BASE_DIR, 'models', 's2v_old')

# Preload models and dataset
random_passage = pd.read_csv(RANDOM_PASSAGE_PATH)

t5ag_model = T5ForConditionalGeneration.from_pretrained(T5AG_MODEL_DIR)
t5ag_tokenizer = T5Tokenizer.from_pretrained(T5AG_TOKENIZER_DIR)
t5qg_model = T5ForConditionalGeneration.from_pretrained(T5QG_MODEL_DIR)
t5qg_tokenizer = T5Tokenizer.from_pretrained(T5QG_TOKENIZER_DIR)
s2v = Sense2Vec().from_disk(S2V_MODEL_PATH)
sentence_transformer_model = SentenceTransformer('sentence-transformers/LaBSE')

def answer_question(question, context):
    """Generate an answer for a given question and context."""
    input_text = f"question: {question} context: {context}"
    input_ids = t5ag_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        output = t5ag_model.generate(input_ids, max_length=512, num_return_sequences=1, max_new_tokens=200)

    return t5ag_tokenizer.decode(output[0], skip_special_tokens=True)


def get_nouns_multipartite(content):
    """Extract keywords from content using MultipartiteRank algorithm."""
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content, language='en')
        pos_tags = {'PROPN', 'NOUN', 'ADJ', 'VERB', 'ADP', 'ADV', 'DET', 'CONJ', 'NUM', 'PRON', 'X'}

        stoplist = list(string.punctuation) + ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')

        extractor.candidate_selection(pos=pos_tags)
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyphrases = extractor.get_n_best(n=15)
        
        return [val[0] for val in keyphrases]
    except Exception as e:
        print(f"Error extracting nouns: {e}")
        return []

    
def get_keywords(passage):

    vectorizer = TfidfVectorizer(stop_words='english')
    
    tfidf_matrix = vectorizer.fit_transform([passage])
    
    feature_names = vectorizer.get_feature_names_out()
    
    tfidf_scores = tfidf_matrix.toarray().flatten()
    
    word_scores = dict(zip(feature_names, tfidf_scores))

    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    keywords = [word for word, score in sorted_words]
    
    return keywords


def get_question(context, answer, model, tokenizer):
    """Generate a question for the given answer and context."""
    answer_span = context.replace(answer, f"<hl>{answer}<hl>") + "</s>"
    inputs = tokenizer(answer_span, return_tensors="pt")
    question = model.generate(input_ids=inputs.input_ids, max_length=50)[0]

    return tokenizer.decode(question, skip_special_tokens=True)


def filter_same_sense_words(original, wordlist):
    """Filter words that have the same sense as the original word."""
    base_sense = original.split('|')[1]
    return [word[0].split('|')[0].replace("_", " ").title().strip() for word in wordlist if word[0].split('|')[1] == base_sense]

def get_max_similarity_score(wordlist, word):
    """Get the maximum similarity score between the word and a list of words."""
    return max(levenshtein.normalized_similarity(word.lower(), each.lower()) for each in wordlist)

def sense2vec_get_words(word, s2v, topn, question):
    """Get similar words using Sense2Vec."""
    try:
        sense = s2v.get_best_sense(word, senses=["NOUN", "PERSON", "PRODUCT", "LOC", "ORG", "EVENT", "NORP", "WORK OF ART", "FAC", "GPE", "NUM", "FACILITY"])
        
        if sense is None:
            print(f"[DEBUG] No suitable sense found for word: '{word}'")
            return []

        most_similar = s2v.most_similar(sense, n=topn)
        output = filter_same_sense_words(sense, most_similar)
    except Exception as e:
        print(f"Error in Sense2Vec: {e}")
        output = []
    
    threshold = 0.6
    final = [word]
    checklist = question.split()

    for similar_word in output:
        if get_max_similarity_score(final, similar_word) < threshold and similar_word not in final and similar_word not in checklist:
            final.append(similar_word)
    
    return final[1:]


def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
    """Maximal Marginal Relevance (MMR) for keyword extraction."""
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        mmr = (lambda_param * candidate_similarities) - ((1 - lambda_param) * target_similarities.reshape(-1, 1))
        mmr_idx = candidates_idx[np.argmax(mmr)]

        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def get_distractors_wordnet(word):
    """Get distractors using WordNet."""
    distractors = []
    try:
        synset = wn.synsets(word, 'n')[0]
        hypernym = synset.hypernyms()
        if not hypernym:
            return distractors
        for item in hypernym[0].hyponyms():
            name = item.lemmas()[0].name().replace("_", " ").title()
            if name.lower() != word.lower() and name not in distractors:
                distractors.append(name)
    except Exception as e:
        print(f"Error in WordNet distractors: {e}")
        pass

    return distractors

def get_distractors(word, original_sentence, sense2vec_model, sentence_model, top_n, lambda_val):
    """Get distractors for a given word using various methods."""
    distractors = sense2vec_get_words(word, sense2vec_model, top_n, original_sentence)
    if not distractors:
        return []

    distractors_new = [word.capitalize()] + distractors
    embedding_sentence = f"{original_sentence} {word.capitalize()}"
    keyword_embedding = sentence_model.encode([embedding_sentence])
    distractor_embeddings = sentence_model.encode(distractors_new)

    max_keywords = min(len(distractors_new), 5)
    filtered_keywords = mmr(keyword_embedding, distractor_embeddings, distractors_new, max_keywords, lambda_val)
    return [kw.capitalize() for kw in filtered_keywords if kw.lower() != word.lower()][1:]


def get_mca_questions(context, qg_model, qg_tokenizer, s2v, sentence_transformer_model, num_questions=5, max_attempts=2):
    """
    Generate multiple-choice answer questions from a given context.
    """
    generated_questions = []

    keywords = get_keywords(context)
    for kw in keywords[:num_questions]:
        attempts = 0
        while attempts < max_attempts:
            question = get_question(context, kw, qg_model, qg_tokenizer)
            if question and question not in [q[0] for q in generated_questions]:
                distractors = get_distractors(kw, context, s2v, sentence_transformer_model, top_n=5, lambda_val=0.5)
                if distractors:
                    generated_questions.append((question, kw, distractors))
                    break
            attempts += 1
    return generated_questions

if __name__ == "__main__":
    # Choose the passage to work with
    context = random_passage.iloc[0]['context']  # or you could use: context = news_passage.iloc[0]['context']

    # Generate multiple-choice questions
    mca_questions = get_mca_questions(context, t5qg_model, t5qg_tokenizer, s2v, sentence_transformer_model, num_questions=5)

    # Print generated questions and distractors
    for i, qa in enumerate(mca_questions, 1):
        print(f"Question {i}: {qa['question']}")
        print(f"Answer: {qa['answer']}")
        print(f"Distractors: {', '.join(qa['distractors'])}\n")
        