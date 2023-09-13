from datasets import load_dataset
import json

# Load the XSum dataset
xsum_dataset = load_dataset("xsum")

# Extract instances from each split
train_subset = xsum_dataset["train"][:100]
val_subset = xsum_dataset["validation"][:10]
test_subset = xsum_dataset["test"][:10]

# Combine the subsets into one dictionary
combined_subset = {
    "train": [example for example in train_subset],
    "validation": [example for example in val_subset],
    "test": [example for example in test_subset]
}

# Save the combined subset to a JSON file
with open("xsum_small_subset.json", "w", encoding="utf-8") as f:
    json.dump(combined_subset, f, ensure_ascii=False, indent=4)




