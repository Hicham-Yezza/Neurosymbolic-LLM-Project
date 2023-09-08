import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, pipeline

# Define a function to preprocess the dataset
def preprocess_data(dataset, max_length=512, batch_size=4):
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    dataset = dataset.map(
        lambda examples: tokenizer(
            examples["document"],
            examples["summary"],
            truncation=True,
            max_length=max_length,
            padding=False,
        ),
        batched=True,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader, tokenizer

# Define a function to fine-tune T5-small
def fine_tune_t5(dataloader, tokenizer, output_dir="t5-small-xsum_20230908_162252", num_train_epochs=1, per_device_train_batch_size=4):
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, max_length=512
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataloader.dataset,
    )
    trainer.train()
    trainer.save_model()

# Define a function to evaluate the model using ROUGE scores
def evaluate_model(dataset, output_dir="t5-small-xsum"):
    summarization = pipeline("summarization", model=output_dir, tokenizer="t5-small")
    actuals = [d["summary"] for d in dataset]
    predictions = [p["summary"] for p in summarization([d["document"] for d in dataset], max_length=150, min_length=10, length_penalty=2.0, num_beams=4)]
    rouge_metric = load_metric("rouge")
    rouge_scores = rouge_metric.compute(predictions=predictions, references=actuals, rouge_types=["rouge1", "rouge2", "rougeL"])
    return rouge_scores

# Add a main block
if __name__ == "__main__":
    # Load the XSum dataset
    xsum_dataset = load_dataset("xsum")
    # Subset the dataset for testing purposes (remove this line for the full dataset)
    xsum_dataset = xsum_dataset["train"].shuffle(seed=42).select([i for i in range(100)])
    # Preprocess the dataset
    dataloader, tokenizer = preprocess_data(xsum_dataset)
    # Fine-tune T5-small
    fine_tune_t5(dataloader, tokenizer)
    # Evaluate the model using ROUGE scores
    rouge_scores = evaluate_model(xsum_dataset)
    # Print ROUGE scores
    for key, value in rouge_scores.items():
        print(f"{key}: {value.mid.fmeasure:.4f}")
