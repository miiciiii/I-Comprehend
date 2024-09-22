import os
import torch
import json
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import nltk
import spacy
import string
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
import numpy as np
import transformers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import T5ForConditionalGeneration, T5TokenizerFast

import warnings
warnings.filterwarnings("ignore")

MODEL3_DIR = os.path.join('..', 'src', 'models', 'qgmodel3')
TOKENIZER_DIR = os.path.join('..', 'src', 'models', 'qgmodel3_tokenizer')
dataset_path = os.path.join('..', 'datasets', 'processed', 'generated_qa.csv')

TOKENIZER = T5TokenizerFast.from_pretrained(TOKENIZER_DIR)
MODEL = T5ForConditionalGeneration.from_pretrained(MODEL3_DIR)
OPTIMIZER = Adam(MODEL.parameters(), lr=0.00001)
Q_LEN = 256   # Question Length
T_LEN = 32    # Target Length
BATCH_SIZE = 4
DEVICE = "cpu"

data = pd.read_csv(dataset_path)
data = data.sample(n=10000).reset_index(drop=True)

class QG_Dataset(Dataset):
    def __init__(self, tokenizer, dataframe, q_len, t_len):
        self.tokenizer = tokenizer
        self.q_len = q_len
        self.t_len = t_len
        self.data = dataframe
        self.questions = self.data["question"]
        self.context = self.data["context"]
        self.answer = self.data['answers']
        self.question_type = self.data['question_type']
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.context[idx]
        answer = self.answer[idx] if isinstance(self.answer[idx], str) else self.answer[idx][0]
        question_type = self.question_type[idx]
        
        context_with_type = f"{question_type} {context}"
        
        question_tokenized = self.tokenizer(
            question, context_with_type, max_length=self.q_len, padding="max_length",
            truncation=True, add_special_tokens=True, return_tensors='pt'
        )
        answer_tokenized = self.tokenizer(
            answer, max_length=self.t_len, padding="max_length", 
            truncation=True, add_special_tokens=True, return_tensors='pt'
        )
        
        labels = answer_tokenized["input_ids"].squeeze()
        labels[labels == 0] = -100
        
        return {
            "input_ids": question_tokenized["input_ids"].squeeze(),
            "attention_mask": question_tokenized["attention_mask"].squeeze(),
            "labels": labels,
            "decoder_attention_mask": answer_tokenized["attention_mask"].squeeze()
        }

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_sampler = RandomSampler(train_data.index)
val_sampler = RandomSampler(val_data.index)

qa_dataset = QG_Dataset(TOKENIZER, data, Q_LEN, T_LEN)

train_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

train_loss = 0
val_loss = 0
train_batch_count = 0
val_batch_count = 0

for epoch in range(2):
    MODEL.train()
    for batch in tqdm(train_loader, desc="Training batches"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        outputs = MODEL(
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          decoder_attention_mask=decoder_attention_mask
                        )

        OPTIMIZER.zero_grad()
        outputs.loss.backward()
        OPTIMIZER.step()
        train_loss += outputs.loss.item()
        train_batch_count += 1
    
    MODEL.eval()
    for batch in tqdm(val_loader, desc="Validation batches"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        outputs = MODEL(
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          decoder_attention_mask=decoder_attention_mask
                        )

        OPTIMIZER.zero_grad()
        outputs.loss.backward()
        OPTIMIZER.step()
        val_loss += outputs.loss.item()
        val_batch_count += 1
        
    print(f"{epoch+1}/{2} -> Train loss: {train_loss / train_batch_count}\tValidation loss: {val_loss/val_batch_count}")
