# 1. Dependencies Installation
# !pip install transformers datasets torch

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset

# 2. Data Loading
def load_xsum_data(subset_size_train=100, subset_size_val=10):
    dataset = load_dataset("xsum")
    # Take only a subset of the dataset
    dataset["train"] = dataset["train"].select(range(subset_size_train))
    dataset["validation"] = dataset["validation"].select(range(subset_size_val))
    return dataset

# 3. Data Preprocessing
def preprocess_data(data, tokenizer):
    def to_t5_format(example):
        return {
            "source_text": example["document"],
            "target_text": example["summary"],
            "input_ids": tokenizer(example["document"], truncation=True, return_tensors="pt").input_ids[0],
            "labels": tokenizer(example["summary"], truncation=True, return_tensors="pt").input_ids[0]
        }
    
    formatted_data = data.map(to_t5_format)
    return formatted_data

# 4. Model and Tokenizer Initialization
def initialize_model_and_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return model, tokenizer

# 5. Training Configuration
# 5. Training Configuration
def get_trainer(model, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir="./t5_xsum_output",  # <-- Add this line to specify output directory
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10, # Reduced for the smaller dataset
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    return trainer

# 6. Model Training
def train_model(trainer):
    trainer.train()

# 7. Evaluation
def evaluate_model(trainer):
    results = trainer.evaluate()
    return results

# 8. Saving the Model
def save_model(model, path="./t5_xsum_model"):
    model.save_pretrained(path)

if __name__ == "__main__":
    # Load Data
    datasets = load_xsum_data()
    
    # Initialize Model and Tokenizer
    model, tokenizer = initialize_model_and_tokenizer()
    
    # Preprocess Data
    train_data = preprocess_data(datasets["train"], tokenizer)
    val_data = preprocess_data(datasets["validation"], tokenizer)
    
    # Get Trainer
    trainer = get_trainer(model, train_data, val_data)
    
    # Train the Model
    train_model(trainer)
    
    # Evaluate the Model
    eval_results = evaluate_model(trainer)
    print(eval_results)
    
    # Save the Model
    save_model(model)
