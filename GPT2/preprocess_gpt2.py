"""
Preprocessing script for GPT-2 model.

This script handles preprocessing of datasets to be compatible with GPT-2 model's requirements.
"""

from transformers import GPT2Tokenizer
from ..preprocessing import *


tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

def preprocess_for_gpt2(dataset):
    """Tokenize data for GPT-2."""
    def tokenize_example(example):
        return tokenizer(example['article'], truncation=True, padding='max_length', max_length=512), tokenizer(example['summary'], truncation=True, padding='max_length', max_length=150)

    return dataset.map(tokenize_example)

# Preprocess datasets
cnn_dailymail = preprocess_for_gpt2(cnn_dailymail)
xsum = preprocess_for_gpt2(xsum)

 
