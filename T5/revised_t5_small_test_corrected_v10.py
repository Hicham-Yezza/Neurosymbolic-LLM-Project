import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

config = {
    "tokenizer_name_or_path": "t5-small",
    "model_name_or_path": "t5-small",
    "batch_size": 4,
    "max_length": 512,
    "num_train_epochs": 1,
    "output_dir": "fine_tuned_T5",
    "eval_steps": 100,
    "logging_steps": 100,
    "save_steps": 1000,
    "per_device_train_batch_size": 4,
}

def preprocess_data(dataset, tokenizer_name_or_instance=, max_length, batch_size):
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


def fine_tune_t5(dataloader, tokenizer, model_name_or_path, output_dir, 
                 num_train_epochs, per_device_train_batch_size, eval_steps, logging_steps, 
                 save_steps, max_length):

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

  # Load the full XSum dataset
  xsum_dataset = load_dataset("xsum") 
  
  # Use the provided training, validation and test splits
  train_dataset = xsum_dataset['train'] 
  val_dataset = xsum_dataset['validation']
  test_dataset = xsum_dataset['test']

   # Preprocess only the training data
    train_dataloader, tokenizer = preprocess_data(train_dataset, 
                                                  config["tokenizer_name_or_path"], 
                                                  config["max_length"], 
                                                  config["batch_size"])

  # Fine-tune model on the training set
    model = fine_tune_t5(train_dataloader, tokenizer, 
                         config["model_name_or_path"], 
                         config["output_dir"], 
                         config["num_train_epochs"], 
                         config["per_device_train_batch_size"], 
                         config["eval_steps"], 
                         config["logging_steps"], 
                         config["save_steps"], 
                         config["max_length"])

  # Evaluate on the validation set during training 
  rouge_scores = evaluate_model(val_dataset, tokenizer, model)
  print("Validation ROUGE Scores:", rouge_scores)

  # Evaluate on the test set after training
  test_rouge_scores = evaluate_model(test_dataset, tokenizer, model)
  print("Test ROUGE Scores:", test_rouge_scores)

if __name__ == "__main__":
    main()
