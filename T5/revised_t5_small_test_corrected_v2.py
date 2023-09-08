import torch
import datasets
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, pipeline
from datetime import datetime

def preprocess_data(dataset, tokenizer, max_length=512, batch_size=4):
    """
    Tokenize the documents and summaries.
    """
    print("Tokenizing data...")
    dataset = dataset.map(
        lambda examples: tokenizer(
            examples["document"],
            examples["summary"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ),
        batched=True,
    )

    return dataset

def fine_tune_t5(dataloader, model_name_or_path="t5-small", output_dir=None, num_train_epochs=3,
                 per_device_train_batch_size=4, eval_steps=500, logging_steps=500, save_steps=500,
                 max_length=512, learning_rate=5e-5):
    """
    Load the model, define the training arguments, and fine-tune the model.
    """
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    print("Defining training arguments...")
    # Define the output directory
    if not output_dir:
        output_dir = f"{model_name_or_path}-finetuned-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Output directory: {output_dir}")

    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        learning_rate=learning_rate,
    )

    print("Creating trainer...")
    # Create the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,  # Ignore loss for padded tokens
            pad_to_multiple_of=8,  # Pad for TPU compatibility
        ),
        train_dataset=dataloader.dataset,
    )

    print("Start training...")
    # Train the model
    trainer.train()

def evaluate_model(dataset, tokenizer, model_path, max_length=150):
    """
    Load the model and evaluate it on the test set.
    """
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    print("Creating summarizer...")
    # Create the summarizer
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    print("Evaluating model...")
    # Evaluate the model
    actual_summaries = [d["summary"] for d in dataset]
    predicted_summaries = summarizer([d["document"] for d in dataset],
                                      max_length=max_length,
                                      min_length=20,
                                      beam_size=4)

    print("Calculating ROUGE scores...")
    # Calculate the ROUGE scores
    rouge_metric = load_metric("rouge")
    rouge_scores = rouge_metric.compute(
        predictions=predicted_summaries,
        references=actual_summaries,
        rouge_types=["rouge1", "rouge2", "rougeL"]
    )

    print("ROUGE scores:")
    print(rouge_scores)

if __name__ == "__main__":
    # Load the dataset
    dataset = datasets.load_dataset("cnn_dailymail", config_name="3.0.0")


    # Split the dataset into train and test sets
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # Preprocess the data
    train_dataloader = preprocess_data(train_dataset, tokenizer)
    test_dataloader = preprocess_data(test_dataset, tokenizer)

    # Fine-tune the model
    fine_tune_t5(train_dataloader, model_name_or_path="t5-small")

    # Evaluate the model
    evaluate_model(test_dataloader, tokenizer, model_path="t5-small-finetuned")