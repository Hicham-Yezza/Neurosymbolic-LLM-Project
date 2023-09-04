"""
Preprocessing script for DistilBERT model.

This script handles preprocessing of datasets to be compatible with DistilBERT model's requirements.
"""

from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_for_distilbert(dataset):
    """Tokenize data for DistilBERT."""
    def tokenize_example(example):
        return tokenizer(example['article'], truncation=True, padding='max_length', max_length=512), tokenizer(example['summary'], truncation=True, padding='max_length', max_length=150)

    return dataset.map(tokenize_example)

# Preprocess datasets
cnn_dailymail = preprocess_for_distilbert(cnn_dailymail)
xsum = preprocess_for_distilbert(xsum)
 
