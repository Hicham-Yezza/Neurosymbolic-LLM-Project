# Necessary libraries and modules for the task
import torch
import wandb
from datasets import load_dataset, load_metric
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer, DataCollatorForSeq2Seq, pipeline)

# Check if CUDA (NVIDIA GPU acceleration) is available and set the device accordingly
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
print(f"Using device: {device}")

def preprocess_data(dataset, tokenizer_name_or_instance="t5-base", max_length=512, batch_size=4):
    # If the tokenizer is provided as a string name, load the tokenizer, otherwise use the provided tokenizer instance
    if isinstance(tokenizer_name_or_instance, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_instance)
    else:
        tokenizer = tokenizer_name_or_instance

    # Tokenize the dataset examples
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples["document"], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        tokenized_labels = tokenizer(examples["summary"], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_labels["input_ids"]
        }

    # Apply the tokenization to the dataset
    dataset = dataset.map(tokenize_function, batched=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader, tokenizer

def fine_tune_t5(dataloader, tokenizer, model_name_or_path="t5-base", output_dir="fine_tuned_T5", 
                 num_train_epochs=1, per_device_train_batch_size=4, eval_steps=100, logging_steps=100, 
                 save_steps=5000, max_length=512):
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    # Define the training arguments for the trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        logging_dir='./logs',
        report_to='wandb'
    )

    # Define the data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, max_length=max_length
    )

    # Initialize the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataloader.dataset,
    )

    # Train the model
    trainer.train()
    # Save the model
    trainer.save_model()
    return model

def evaluate_model(dataset, tokenizer, model):
    # Ensure that a model is provided
    assert model is not None, "The model passed to evaluate_model is None!"

    # Extract the actual summaries
    actuals = [d["summary"] for d in dataset]
    # Tokenize the documents
    inputs = tokenizer([d["document"] for d in dataset], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    # Generate summaries using the model
    outputs = model.generate(**inputs, max_length=150, min_length=10, length_penalty=2.0, num_beams=4)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Compute the ROUGE scores
    rouge_metric = load_metric("rouge")
    rouge_scores = rouge_metric.compute(predictions=predictions, references=actuals, rouge_types=["rouge1", "rouge2", "rougeL"])
    
    # Log the ROUGE scores to wandb
    wandb.log(rouge_scores)

    return rouge_scores

def main():
    # Initialize wandb
    wandb.init(project="t5_finetuning", name="t5_on_xsum")

    # Load the XSum dataset
    xsum_dataset = load_dataset("xsum")
    # Subset the dataset for testing purposes (can be removed for the full dataset)
    xsum_dataset = xsum_dataset["train"].shuffle(seed=42).select([i for i in range(100)])

    # Preprocess the dataset
    dataloader, tokenizer = preprocess_data(xsum_dataset)

    # Fine-tune the T5 model
    model = fine_tune_t5(dataloader, tokenizer)

    # Evaluate the model and compute ROUGE scores
    rouge_scores = evaluate_model(xsum_dataset, tokenizer, model)
    for key, value in rouge_scores.items():
        print(f"{key}: {value.mid.fmeasure:.4f}")

    # Finish the wandb run
    wandb.finish()

# Execute the main function if the script is run as the main module
if __name__ == "__main__":
    main()
