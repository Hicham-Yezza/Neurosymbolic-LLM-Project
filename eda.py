# eda.py - Utility script for exploratory data analysis

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Import the load_cnn_dailymail and load_xsum functions from load_datasets.py
from load_datasets import load_cnn_dailymail, load_xsum

# Loading the CNN/Daily Mail and XSum datasets
cnn_dailymail = load_cnn_dailymail()
xsum = load_xsum()

# Printing basic information about the loaded datasets, such as their names and versions
print("CNN/Daily Mail dataset:")
print(cnn_dailymail)
print("\nXSum dataset:")
print(xsum)

# Accessing the "train" split of both the CNN/Daily Mail and XSum datasets
train_cnn_dailymail = cnn_dailymail["train"]
train_xsum = xsum["train"]

# Print the number of examples in the "train" split of each dataset
print("\nNumber of examples in CNN/Daily Mail train split:", len(train_cnn_dailymail))
print("Number of examples in XSum train split:", len(train_xsum))

# Accessing example texts and summaries from the first example in each dataset
example_text_cnn_dailymail = train_cnn_dailymail["article"][0]
example_summary_cnn_dailymail = train_cnn_dailymail["highlights"][0]
example_text_xsum = train_xsum["document"][0]
example_summary_xsum = train_xsum["summary"][0]

# Printing example texts and summaries for both datasets
print("\nExample text (CNN/Daily Mail):")
print(example_text_cnn_dailymail)
print("\nExample summary (CNN/Daily Mail):")
print(example_summary_cnn_dailymail)
print("\nExample text (XSum):")
print(example_text_xsum)
print("\nExample summary (XSum):")
print(example_summary_xsum)

# Performing exploratory data analysis (EDA)
# Loading data from the datasets and analyzing it
# Calculating and visualize summary lengths
summary_lengths_cnn_dailymail = [len(summary.split()) for summary in train_cnn_dailymail["highlights"]]
summary_lengths_xsum = [len(summary.split()) for summary in train_xsum["summary"]]

# Plot analysis results
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.hist(summary_lengths_cnn_dailymail, bins=20, range=(0, 200), edgecolor="k", alpha=0.7)
plt.xlabel("Summary Length")
plt.ylabel("Frequency")
plt.title("Distribution of Summary Lengths (CNN/Daily Mail)")

plt.subplot(2, 1, 2)
plt.hist(summary_lengths_xsum, bins=20, range=(0, 50), edgecolor="k", alpha=0.7)
plt.xlabel("Summary Length")
plt.ylabel("Frequency")
plt.title("Distribution of Summary Lengths (XSum)")

plt.tight_layout()
plt.show()
