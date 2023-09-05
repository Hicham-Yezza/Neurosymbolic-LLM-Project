# load_datasets.py

# Required Libraries

from datasets import load_dataset

def load_cnn_dailymail():
    """
    Load the CNN/Daily Mail dataset.
    
    Dataset Stats:
    - Number of articles: ~300,000
    - Average article length: Varies but typically a few hundred to over a thousand words.
    - Summaries: Abstractive
    
    Returns:
        DatasetDict: HuggingFace's dataset object containing train, test, and validation splits.
    """
    return load_dataset("cnn_dailymail", "3.0.0")

def load_xsum():
    """
    Load the XSum dataset.
    
    Dataset Stats:
    - Number of articles: ~230,000
    - Average article length: Varies, typically shorter than CNN/Daily Mail.
    - Summaries: Extremely abstractive and concise, typically a single sentence.
    
    Returns:
        DatasetDict: HuggingFace's dataset object containing train, test, and validation splits.
    """
    return load_dataset("xsum")

if __name__ == "__main__":
    # Load datasets
    cnn_dailymail = load_cnn_dailymail()
    xsum = load_xsum()

    # Print success messages
    print("CNN/Daily Mail dataset loaded successfully.")
    print("XSum dataset loaded successfully.")
