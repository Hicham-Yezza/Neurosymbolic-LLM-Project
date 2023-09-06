import re
import logging
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    try:
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

    except Exception as e:
        logger.error(f"Error during text cleaning: {e}")
        return None

    return text.strip()

def tokenize_sentences(text):
    try:
        return sent_tokenize(text)
    except Exception as e:
        logger.error(f"Error during sentence tokenization: {e}")
        return []

def tokenize_words(text):
    try:
        return word_tokenize(text)
    except Exception as e:
        logger.error(f"Error during word tokenization: {e}")
        return []

def conditional_dataset_split(dataset, train_split=0.8, val_split=0.1):
    try:
        if "train" in dataset:
            return dataset["train"], dataset.get("validation", None), dataset.get("test", None)

        # Split dataset
        data_len = len(dataset)
        train_len = int(data_len * train_split)
        val_len = int(data_len * val_split)

        train_data = dataset.select(range(train_len))
        val_data = dataset.select(range(train_len, train_len + val_len))
        test_data = dataset.select(range(train_len + val_len, data_len))

        return train_data, val_data, test_data

    except Exception as e:
        logger.error(f"Error during dataset split: {e}")
        return None, None, None

def reduce_dataset_size(dataset, fraction=0.05):
    try:
        reduced_len = int(len(dataset) * fraction)
        return dataset.select(range(reduced_len))
    except Exception as e:
        logger.error(f"Error during dataset size reduction: {e}")
        return dataset

def demonstrate_preprocessing():
    try:
        # Load the IMDb dataset
        imdb_dataset = load_dataset("imdb")
    except Exception as e:
        logger.error(f"Error loading IMDb dataset: {e}")
        return


    imdb_review = imdb_dataset["train"][0]["text"]

    # Clean and tokenize the IMDb review
    imdb_cleaned = clean_text(imdb_review)
    imdb_sentences = tokenize_sentences(imdb_cleaned)
    imdb_words = tokenize_words(imdb_cleaned)

    # Print and assert IMDb results
    logger.info("Processed IMDb Review.")
    print(imdb_cleaned)
    print(imdb_sentences)
    print(imdb_words)

    # Load the CNN/DailyMail dataset
    cnn_dailymail_dataset = load_dataset("cnn_dailymail", "3.0.0")
    cnn_dailymail_article = cnn_dailymail_dataset["train"][0]["article"]

    # Clean and tokenize the CNN/DailyMail article
    cnn_dailymail_cleaned = clean_text(cnn_dailymail_article)
    cnn_dailymail_sentences = tokenize_sentences(cnn_dailymail_cleaned)
    cnn_dailymail_words = tokenize_words(cnn_dailymail_cleaned)

    # Print and assert CNN/DailyMail results
    logger.info("Processed CNN/DailyMail Article.")
    print(cnn_dailymail_cleaned)
    print(cnn_dailymail_sentences)
    print(cnn_dailymail_words)

    # Load the XSum dataset
    xsum_dataset = load_dataset("xsum")
    xsum_document = xsum_dataset["train"][0]["document"]

    # Clean and tokenize the XSum document
    xsum_cleaned = clean_text(xsum_document)
    xsum_sentences = tokenize_sentences(xsum_cleaned)
    xsum_words = tokenize_words(xsum_cleaned)

    # Print and assert XSum results
    logger.info("Processed XSum Document.")
    print(xsum_cleaned)
    print(xsum_sentences)
    print(xsum_words)

if __name__ == "__main__":
    demonstrate_preprocessing()
