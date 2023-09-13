import torch
import wandb
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, pipeline
from tqdm import tqdm  # Importing tqdm for progress bars

def initialize_device():
    use_cuda = torch.cuda.is_available()
    return "cuda" if use_cuda else "cpu"

def initialize_wandb():
    return wandb.init(project="t5-small-xsum")
    
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


def fine_tune_t5(dataloader, tokenizer, device, model_name_or_path="t5-small", output_dir="fine_tuned_T5", 
                 num_train_epochs=1, per_device_train_batch_size=4, eval_steps=100, logging_steps=100, 
                 save_steps=1000, max_length=512):

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model.to(device)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        report_to="wandb"
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
    
def evaluate_model(dataset, tokenizer, model, batch_size=4):
    """
    Evaluate the trained model using ROUGE metrics.
    
    Parameters:
    - dataset (Dataset): The dataset to evaluate the model on.
    - tokenizer (AutoTokenizer): The tokenizer instance to process text.
    - model (AutoModelForSeq2SeqLM): The trained model for evaluation.
    - batch_size (int): Number of samples per batch for evaluation.
    
    Returns:
    - rouge_scores (dict): A dictionary containing calculated ROUGE scores.
    """

    # Assert that the model instance exists.
    assert model is not None, "The model passed to evaluate_model is None!"

    # Initialize the ROUGE metric from Hugging Face.
    rouge_metric = load_metric("rouge")

    # Initialize lists to store the actual summaries and the generated summaries.
    actuals = []
    predictions = []

    # Loop through the dataset in batches to prevent memory overflow.
    for i in range(0, len(dataset), batch_size):

        # Create a batch from the dataset.
        batch = dataset[i: i + batch_size]
        
        # Append the actual summaries to the list.
        actuals.extend([d["summary"] for d in batch])

        # Tokenize the documents in the current batch.
        inputs = tokenizer([d["document"] for d in batch], return_tensors="pt", truncation=True, padding="max_length", max_length=512)

        # Generate summaries using the model.
        outputs = model.generate(**inputs, max_length=150, min_length=10, length_penalty=2.0, num_beams=4)

        # Decode the generated summaries.
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Append the generated summaries to the list.
        predictions.extend(decoded_preds)

    # Compute the ROUGE scores based on the actual and generated summaries.
    rouge_scores = rouge_metric.compute(predictions=predictions, references=actuals, rouge_types=["rouge1", "rouge2", "rougeL"])

    return rouge_scores


def log_metrics(rouge_scores):
    # Log metrics to console
    for key, value in rouge_scores.items():
        fmeasure_value = value.mid.fmeasure
        precision_value = value.mid.precision
        recall_value = value.mid.recall
        print(f"{key}: F-measure: {fmeasure_value:.4f}, Precision: {precision_value:.4f}, Recall: {recall_value:.4f}")

    # Prepare metrics for Wandb logging
    metrics_to_log = {}
    for key, value in rouge_scores.items():
        metrics_to_log[f"{key}_fmeasure"] = value.mid.fmeasure
        metrics_to_log[f"{key}_precision"] = value.mid.precision
        metrics_to_log[f"{key}_recall"] = value.mid.recall
    
    # Log metrics to Weights and Biases
    wandb.log(metrics_to_log)


def main():
    # Initialize device and WandB
    device = initialize_device()
    wandb_run = initialize_wandb()

    print(f"Using device: {device}")  

    # Load the XSum dataset
    xsum_dataset = load_dataset("xsum")
    
    # Get different splits
    xsum_train_dataset = xsum_dataset["train"].shuffle(seed=42).select([i for i in range(100)])  # Subset for testing
    xsum_eval_dataset = xsum_dataset["validation"].shuffle(seed=42).select([i for i in range(50)])  # Subset for testing

    # Preprocess the dataset
    train_dataloader, tokenizer = preprocess_data(xsum_train_dataset)

    # Fine-tune T5-small
    model = fine_tune_t5(train_dataloader, tokenizer, device)
    
    # Evaluate the model using ROUGE scores on a different dataset split
    rouge_scores = evaluate_model(xsum_eval_dataset, tokenizer, model)

    # Log wanbd metrics
    log_metrics(rouge_scores)
    
    # Close WandB logging 
    wandb_run.finish()

if __name__ == "__main__":
    main()



