import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import torch

TRAINING_CSV="data/processed/reddit/labeled_with_generic_comments.csv"
OUTPUT_DIR = "models/teacher_4class_sample_weighted"
MODEL_NAME = "roberta-base"

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=160,
    )

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    metrics = {}
    metrics.update(accuracy.compute(predictions=preds, references=labels))
    metrics.update(
        f1.compute(predictions=preds, references=labels, average="macro")
    )
    return metrics

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        sample_weight = inputs.pop("sample_weight", None)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        if sample_weight is not None:
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_example_loss = loss_fct(logits, labels)
            
            weights = sample_weight.to(per_example_loss.device)
            weighted_loss = (per_example_loss * weights).mean()
            loss = weighted_loss

        return (loss, outputs) if return_outputs else loss


def add_context(example):
    old_t = example.get("old_team", "")
    new_t = example.get("new_team", "")
    txt = example["text"]
    example["text"] = f"{txt} [SEP] {old_t} [SEP] {new_t}"
    return example

if __name__ == "__main__":

    df = pd.read_csv(TRAINING_CSV)

    df["label"] = df["label"].astype(int)

    tmp_csv = "data/processed/reddit/_tmp_weighted_train.csv"
    df.to_csv(tmp_csv, index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_files = {"train": tmp_csv}
    raw_datasets = load_dataset("csv", data_files=data_files)
    raw_datasets = raw_datasets.map(add_context)

    tokenized = raw_datasets.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    train_dataset = tokenized["train"].remove_columns(
        [c for c in tokenized["train"].column_names
        if c not in ["input_ids", "attention_mask", "label", "text", "sample_weight"]]
    )
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch")

    splits = train_dataset.train_test_split(test_size=0.2, seed=42)
    train_split = splits["train"]
    val_split = splits["test"]

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        warmup_ratio=0.1
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = WeightedTrainer( 
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=val_split,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    metrics = trainer.evaluate()
    print(metrics)
