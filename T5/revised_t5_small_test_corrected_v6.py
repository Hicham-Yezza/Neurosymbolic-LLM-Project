import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, pipeline
import os
from datetime import datetime

def generate_output_dir(subfolder="fine_tuned_models"):
    """
    Generate an output directory path using the current timestamp.
    
    Args:
    - subfolder (str): The subfolder within the current directory where the models will be saved.

    Returns:
    - str: A directory path with the format 'current_directory/subfolder/model_TIMESTAMP'.
    """
    base_dir = os.getcwd()  # Get the current working directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(base_dir, subfolder, f"model_{timestamp}")


def preprocess_data(dataset, tokenizer_name_or_instance="t5-small", max_length=512, batch_size=4):
    if isinstance(tokenizer_name_or_instance, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_instance)
    else:
        tokenizer = tokenizer_name_or_instance

    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples["document"], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        tokenized_labels = tokenizer(examples["summary"], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        tokenized = {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_labels["input_ids"]
        }
        return tokenized

    dataset = dataset.map(tokenize_function, batched=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader, tokenizer

def fine_tune_t5(dataloader, tokenizer, model_name_or_path="t5-small", output_dir=generate_output_dir(), 
                 num_train_epochs=1, per_device_train_batch_size=4, eval_steps=100, logging_steps=100, 
                 save_steps=1000, max_length=512):
                 
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, max_length=max_length
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataloader.dataset,
    )
    trainer.train()
    trainer.save_model()
    return model

def evaluate_model(dataset, tokenizer, model):
    assert model is not None, "The model passed to evaluate_model is None!"
    actuals = [d["summary"] for d in dataset]
    inputs = tokenizer([d["document"] for d in dataset], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    outputs = model.generate(**inputs, max_length=150, min_length=10, length_penalty=2.0, num_beams=4)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    rouge_metric = load_metric("rouge")
    rouge_scores = rouge_metric.compute(predictions=predictions, references=actuals, rouge_types=["rouge1", "rouge2", "rougeL"])
    return rouge_scores


def main():
    # Load the XSum dataset
    xsum_dataset = load_dataset("xsum")
    # Subset the dataset for testing purposes (remove this line for the full dataset)
    xsum_dataset = xsum_dataset["train"].shuffle(seed=42).select([i for i in range(100)])
    # Preprocess the dataset
    dataloader, tokenizer = preprocess_data(xsum_dataset)
    # Fine-tune T5-small
    model = fine_tune_t5(dataloader, tokenizer)
    # Evaluate the model using ROUGE scores
    rouge_scores = evaluate_model(xsum_dataset, tokenizer, model)
    # Print ROUGE scores
    for key, value in rouge_scores.items():
        print(f"{key}: {value.mid.fmeasure:.4f}")

if __name__ == "__main__":
    main()
