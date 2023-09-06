# preprocessing.py - Utility script for common preprocessing functions.

# Import necessary libraries
import re
from nltk.tokenize import sent_tokenize, word_tokenize

# Define function to tokenize text into sentences
def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

# Define function to tokenize text into words
def tokenize_words(text):
    words = word_tokenize(text)
    return words

# Main code to demonstrate preprocessing functions
if __name__ == "__main__":
    input_text = "Sample text for preprocessing."
    sentences = tokenize_sentences(input_text)
    words = tokenize_words(input_text)
    print("Sentences:", sentences)
    print("Words:", words)
