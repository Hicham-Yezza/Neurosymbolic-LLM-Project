from transformers import DistilBertTokenizer
from datasets import load_dataset, DatasetDict
import os

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

DATASET_CONFIG = {
    'cnn_dailymail': {'version': '3.0.0', 'article_key': 'article', 'summary_key': 'highlights'},
    'xsum': {'version': None, 'article_key': 'document', 'summary_key': 'summary'}
}

def preprocess_for_distilbert(dataset, article_key, summary_key):
    """Tokenize data for DistilBERT."""
    def tokenize_example(example):
        return {
            'input_article': tokenizer(example[article_key], truncation=True, padding='max_length', max_length=512),
            'input_summary': tokenizer(example[summary_key], truncation=True, padding='max_length', max_length=150)
        }
    return dataset.map(tokenize_example)

def main():
    datasets = {}
    processed_datasets = {}

    # Ensure there's a directory to save the processed data
    if not os.path.exists('./processed_data'):
        os.makedirs('./processed_data')

    for dataset_name, config in DATASET_CONFIG.items():
        try:
            if config['version']:
                dataset = load_dataset(dataset_name, config['version'])
            else:
                dataset = load_dataset(dataset_name)
            
            if dataset:
                datasets[dataset_name] = dataset['train']
                processed_dataset = preprocess_for_distilbert(dataset['train'], config['article_key'], config['summary_key'])
                processed_datasets[dataset_name] = processed_dataset
                processed_dataset.save_to_disk(f"./processed_data/{dataset_name}")

        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")

if __name__ == "__main__":
    main()
