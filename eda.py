# eda.py - Utility script for exploratory data analysis

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from scipy import stats

# Configuration for the histograms
HISTOGRAM_CONFIG = {
    'cnn_dailymail': {'bins': 40, 'range': (0, 1300)},
    'xsum': {'bins': 20, 'range': (0, 80)}
}

# Configuration for datasets and their keys
DATASET_CONFIG = {
    'cnn_dailymail': {'version': '3.0.0', 'article_key': 'article', 'summary_key': 'highlights'},
    'xsum': {'version': None, 'article_key': 'document', 'summary_key': 'summary'}
}

SAMPLE_SIZE = 10000  # Default sample size

def load_and_print_dataset_info(dataset_name, version=None):
    try:
        if version:
            dataset = load_dataset(dataset_name, version)
        else:
            dataset = load_dataset(dataset_name)
        
        print(f"\n{dataset_name} dataset:")
        for split, data in dataset.items():
            print(f"{split} size: {len(data)}")
        return dataset
    except Exception as e:
        print(f"Error loading {dataset_name} dataset. Check dataset name, version, or network connection. Detailed Error: {e}")
        return None

def print_example_texts_and_summaries(dataset, article_key, summary_key, num_examples=3):
    for i in range(num_examples):
        example_text = dataset[article_key][i]
        example_summary = dataset[summary_key][i]
        print(f"\nExample text {i+1}:")
        print(example_text)
        print(f"\nExample summary {i+1}:")
        print(example_summary)

def calculate_summary_lengths(dataset, summary_key):
    return [len(summary.split()) for summary in dataset[summary_key]]

def display_statistics(data, dataset_name):
    mean = np.mean(data)
    median = np.median(data)
    mode = stats.mode(data).mode[0]
    print(f"\nStatistics for {dataset_name}:")
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")

def draw_mean_median_lines(data):
    """Helper function to draw mean and median lines on a histogram."""
    mean = np.mean(data)
    median = np.median(data)
    plt.axvline(mean, color='r', linestyle='--')
    plt.axvline(median, color='g', linestyle='-')
    plt.legend({'Mean': mean, 'Median': median})

def plot_summary_lengths_histogram(data, title, dataset_name, subplot_position):
    config = HISTOGRAM_CONFIG.get(dataset_name)
    plt.subplot(*subplot_position)
    plt.hist(data, bins=config['bins'], range=config['range'], edgecolor="k", alpha=0.7)
    draw_mean_median_lines(data)  # Use the helper function
    plt.xlabel("Summary Length")
    plt.ylabel("Frequency")
    plt.title(title)

if __name__ == "__main__":
    # Using the dataset configurations to manage dataset-specific attributes.
    datasets = {}
    for dataset_name, config in DATASET_CONFIG.items():
        dataset = load_and_print_dataset_info(dataset_name, config['version'])
        if dataset:
            datasets[dataset_name] = dataset

    if 'cnn_dailymail' in datasets:
        train_cnn_dailymail = datasets['cnn_dailymail'].get('train')
        if train_cnn_dailymail:
            print("\nCNN/Daily Mail examples:")
            print_example_texts_and_summaries(train_cnn_dailymail, DATASET_CONFIG['cnn_dailymail']['article_key'], DATASET_CONFIG['cnn_dailymail']['summary_key'])

    if 'xsum' in datasets:
        train_xsum = datasets['xsum'].get('train')
        if train_xsum:
            print("\nXSum examples:")
            print_example_texts_and_summaries(train_xsum, DATASET_CONFIG['xsum']['article_key'], DATASET_CONFIG['xsum']['summary_key'])
    
    summary_lengths = {}
    for dataset_name, dataset_data in datasets.items():
        train_data = dataset_data.get('train')
        if train_data:
            lengths = calculate_summary_lengths(train_data, DATASET_CONFIG[dataset_name]['summary_key'])
            summary_lengths[dataset_name] = lengths
            display_statistics(lengths, dataset_name)
    
    # Plot histograms for summary lengths of each dataset
    plt.figure(figsize=(10, 6))
    for index, (dataset_name, lengths) in enumerate(summary_lengths.items(), 1):
        plot_summary_lengths_histogram(lengths, f"Distribution of Summary Lengths ({dataset_name})", dataset_name, (2, 1, index))
    plt.tight_layout()
    plt.show()

