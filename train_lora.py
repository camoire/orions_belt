import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

BASE_MODEL = "/models/qwen_model"
DATASET_PATH = "dataset/train.jsonl"
VAL_PATH = "dataset/val.jsonl"
OUTPUT_DIR = "lora_adapter"

def train_lora_adapter():
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={"train": DATASET_PATH, "validation": VAL_PATH})

    print("Loading model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        local_files_only=True
    )

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    def tokenize(batch):
        full_text = [
            p + r
            for p, r in zip(batch["prompt"], batch["response"])
        ]

        tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=2048,
            padding="max_length",
        )

        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        max_steps=500,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"]
    )

    print("Training LoRA adapter...")
    trainer.train()

    print("Saving LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR)
    print("Training complete.")

if __name__ == "__main__":
    train_lora_adapter()
