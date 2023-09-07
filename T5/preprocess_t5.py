"""
Preprocessing script for T5 model.

This script handles preprocessing of datasets to be compatible with T5 model's requirements.
"""

from transformers import T5Tokenizer
from ..preprocessing import *


def preprocess_for_t5(dataset):
    """Preprocess data for T5."""
    def format_example(example):
        return {
            "input_text": f"summarize: {example['article']}",
            "target_text": example['summary']
        }
    return dataset.map(format_example)

# Preprocess datasets
cnn_dailymail = preprocess_for_t5(cnn_dailymail)
xsum = preprocess_for_t5(xsum)
