import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, pipeline

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


def fine_tune_t5(dataloader, tokenizer, model_name_or_path="t5-small", output_dir="t5-small-xsum_20230908_162252", 
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

def evaluate_model(dataset, tokenizer, model_output_dir="T5/fine_tuned_T5"):
    summarization = pipeline("summarization", model=model_output_dir, tokenizer=tokenizer)
    actuals = [d["summary"] for d in dataset]
    predictions = [p["summary"] for p in summarization([d["document"] for d in dataset], max_length=150, min_length=10, length_penalty=2.0, num_beams=4)]
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
    fine_tune_t5(dataloader, tokenizer)
    # Evaluate the model using ROUGE scores
    rouge_scores = evaluate_model(xsum_dataset, tokenizer)
    # Print ROUGE scores
    for key, value in rouge_scores.items():
        print(f"{key}: {value.mid.fmeasure:.4f}")

if __name__ == "__main__":
    main()
