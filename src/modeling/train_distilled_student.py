import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
import torch.nn.functional as F
import evaluate

INPUT_CSV = "data/processed/reddit/_tmp_distill.csv"
TEACHER_LOGITS_NPY = "data/processed/reddit/teacher_logits.npy"
STUDENT_MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/student_distilled"
NUM_LABELS = 4

teacher_logits = np.load(TEACHER_LOGITS_NPY)

class DistillationTrainer(Trainer):
    def __init__(self, teacher_logits, temperature=2.0, alpha=0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_logits = teacher_logits
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        print(inputs.keys())
        labels = inputs.pop("labels")     
        idx = inputs.pop("idx")      
        outputs = model(**inputs)
        student_logits = outputs.logits

        ce_loss = torch.nn.functional.cross_entropy(student_logits, labels)

        with torch.no_grad():
            t_logits = torch.tensor(
                self.teacher_logits[idx.cpu().numpy()],
                dtype=student_logits.dtype,
                device=student_logits.device,
            )

        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(t_logits / T, dim=-1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)

        loss = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss
        return (loss, outputs) if return_outputs else loss

def tokenize_fn(batch, tokenizer):
    if "_idx" not in batch:
        print("_idx missing from batch")
        print()
        print()
    else:
        print("batch _idx:", batch["_idx"][:5])
        print()
        print()
    enc = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=160,
    )
    enc["labels"] = batch["label"]
    enc["idx"] = batch["_idx"]
    return enc

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics = {}
    metrics.update(accuracy.compute(predictions=preds, references=labels))
    metrics.update(f1.compute(predictions=preds, references=labels, average="macro"))
    return metrics

def main():
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)
    raw = load_dataset("csv", data_files={"train": INPUT_CSV})["train"]
    tokenized = raw.map(lambda b: tokenize_fn(b, tokenizer), batched=True)
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "idx"],
    )


    splits = tokenized.train_test_split(test_size=0.2, seed=42)
    train_ds, val_ds = splits["train"], splits["test"]

    student = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL_NAME, num_labels=NUM_LABELS
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_strategy="epoch",
        remove_unused_columns=False,
    )

    trainer = DistillationTrainer(
        teacher_logits=teacher_logits,
        temperature=2.0,
        alpha=0.7,
        model=student,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
