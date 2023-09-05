# eda.py - Utility script for exploratory data analysis

# Import necessary libraries
import matplotlib.pyplot as plt
from datasets import load_dataset

# Configuration for the histograms
# This makes it easy to adjust settings for each dataset
HISTOGRAM_CONFIG = {
    'cnn_dailymail': {'bins': 40, 'range': (0, 1300)},
    'xsum': {'bins': 20, 'range': (0, 80)}
}

def load_and_print_dataset_info(dataset_name):
    """
    Load a dataset and print its basic information.
    
    Parameters:
        dataset_name (str): Name of the dataset to be loaded.

    Returns:
        DatasetDict: The dataset loaded using HuggingFace's datasets library.
    """
    try:
        dataset = load_dataset(dataset_name)
        print(f"\n{dataset_name} dataset:")
        # Loop through each split (train, test, validation) and print its size
        for split, data in dataset.items():
            print(f"{split} size: {len(data)}")
        return dataset
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {e}")
        return None

def print_example_texts_and_summaries(dataset, article_key, summary_key):
    """
    Print example texts and summaries from a dataset.
    
    Parameters:
        dataset (DatasetDict): The dataset from which to extract example texts and summaries.
        article_key (str): The key to access the articles in the dataset.
        summary_key (str): The key to access the summaries in the dataset.
    """
    example_text = dataset[article_key][0]
    example_summary = dataset[summary_key][0]

    print("\nExample text:")
    print(example_text)
    print("\nExample summary:")
    print(example_summary)

def calculate_summary_lengths(dataset, summary_key):
    """
    Calculate lengths of summaries based on word count.
    """
    return [len(summary.split()) for summary in dataset[summary_key]]

def plot_summary_lengths_histogram(data, title, dataset_name, subplot_position):
    """
    Plot histogram for summary lengths based on provided configuration.
    """
    config = HISTOGRAM_CONFIG.get(dataset_name)
    plt.subplot(*subplot_position)
    plt.hist(data, bins=config['bins'], range=config['range'], edgecolor="k", alpha=0.7)
    plt.xlabel("Summary Length")
    plt.ylabel("Frequency")
    plt.title(title)

if __name__ == "__main__":
    # Load datasets using the helper function
    cnn_dailymail = load_and_print_dataset_info("cnn_dailymail")
    xsum = load_and_print_dataset_info("xsum")

    # Accessing the "train" split for further analysis
    train_cnn_dailymail = cnn_dailymail["train"]
    train_xsum = xsum["train"]

    # Print example articles and summaries for both datasets
    print("\nCNN/Daily Mail examples:")
    print_example_texts_and_summaries(train_cnn_dailymail, "article", "highlights")
    print("\nXSum examples:")
    print_example_texts_and_summaries(train_xsum, "document", "summary")

    # Calculate summary lengths using the helper function
    summary_lengths_cnn_dailymail = calculate_summary_lengths(train_cnn_dailymail, "highlights")
    summary_lengths_xsum = calculate_summary_lengths(train_xsum, "summary")

    # Plot histograms for summary lengths of each dataset
    plt.figure(figsize=(10, 6))
    plot_summary_lengths_histogram(summary_lengths_cnn_dailymail, "Distribution of Summary Lengths (CNN/Daily Mail)", 'cnn_dailymail', (2, 1, 1))
    plot_summary_lengths_histogram(summary_lengths_xsum, "Distribution of Summary Lengths (XSum)", 'xsum', (2, 1, 2))
    plt.tight_layout()
    plt.show()
