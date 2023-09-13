import torch
import wandb
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# Initialize Weights and Biases for experiment tracking
def initialize_wandb():
    """Initializes the Weights and Biases project."""
    return wandb.init(project="t5-small-xsum")

# Preprocess the data and tokenize using a given tokenizer
def preprocess_data(dataset, tokenizer_name_or_instance="t5-small", max_length=512, batch_size=4):
    """Preprocesses the dataset and returns a DataLoader and tokenizer."""
    if isinstance(tokenizer_name_or_instance, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_instance)
    else:
        tokenizer = tokenizer_name_or_instance

    # Function to tokenize the dataset
    def tokenize_function(examples):
        """Tokenizes the text and adds 'input_ids', 'attention_mask', and 'labels' to the dataset."""
        inputs = tokenizer(examples["document"], truncation=True, padding='max_length', max_length=max_length)
        labels = tokenizer(examples["summary"], truncation=True, padding='max_length', max_length=max_length)
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"]
        }

    dataset = dataset.map(tokenize_function, batched=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader, tokenizer

# Fine-tuning function for T5 model
def fine_tune_t5(dataloader, tokenizer, model_name_or_path="t5-small", output_dir="fine_tuned_T5"):
    """Fine-tunes the T5 model."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=1
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataloader.dataset,
    )
    trainer.train()
    trainer.save_model()
    return model    

# Model evaluation function

def evaluate_model(dataset, tokenizer, model):
    assert model is not None, "The model passed to evaluate_model is None!"
    actuals = [d["summary"] for d in dataset]
    inputs = tokenizer([d["document"] for d in dataset], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    outputs = model.generate(**inputs, max_length=150, min_length=10, length_penalty=2.0, num_beams=4)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    rouge_metric = load_metric("rouge")
    rouge_scores = rouge_metric.compute(predictions=predictions, references=actuals, rouge_types=["rouge1", "rouge2", "rougeL"])
    return rouge_scores

# Main function
def main():
    """Main function to run the pipeline."""
    wandb_run = initialize_wandb()

    xsum_dataset = load_dataset("xsum")
    xsum_train_dataset = xsum_dataset["train"].shuffle(seed=42).select(range(100))
    xsum_eval_dataset = xsum_dataset["validation"].shuffle(seed=42).select(range(50))

    train_dataloader, tokenizer = preprocess_data(xsum_train_dataset)
    model = fine_tune_t5(train_dataloader, tokenizer)
    
    eval_dataloader, _ = preprocess_data(xsum_eval_dataset)
    rouge_scores = evaluate_model(eval_dataloader, tokenizer, model)

    wandb.log(rouge_scores)
    wandb_run.finish()

if __name__ == "__main__":
    main()
