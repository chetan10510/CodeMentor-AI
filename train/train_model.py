import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

# Config
model_name = "google/flan-t5-small"
data_path = "data/final_coding_dataset.jsonl"

# Load dataset
dataset = load_dataset("json", data_files=data_path, split="train")

# Format data for T5
def format_example(example):
    return {
        "input_text": f"Question: {example['prompt']}",
        "target_text": example["completion"]
    }

dataset = dataset.map(format_example)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    input_enc = tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=512)
    target_enc = tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=128)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Training args
training_args = TrainingArguments(
    output_dir="model/codementor-flan",
    num_train_epochs=6,                      #  use epochs here
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    report_to="none",
    fp16=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Train
trainer.train()

# Save final model
model.save_pretrained("model/codementor-flan")
tokenizer.save_pretrained("model/codementor-flan")
