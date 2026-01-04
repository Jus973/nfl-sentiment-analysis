import numpy as np
import pandas as pd
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import torch
from torch import nn
from transformers import AutoModel, AutoConfig

CSV_PATH = "data/processed/reddit/full_pre_model_comment_set.csv"
MODEL_NAME = "roberta-large"
MAX_LENGTH = 160
NUM_LABELS = 4

def preprocess_function(examples, tokenizer):
    texts = [
        (
            f"PLAYER: {p} | "
            f"GIVING: {ot} | "
            f"RECEIVING: {nt} | "
            f"COMMENT: {t}"
        )
        for p, ot, nt, t in zip(
            examples["player_name"],
            examples["old_team"],
            examples["new_team"],
            examples["text"],
        )
    ]

    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    aux = np.stack(
        [
            examples["lex_pos"],
            examples["lex_neu"],
            examples["lex_neg"],
            examples["lex_relevance"],
            examples["sent_neg"],
            examples["sent_neu"],
            examples["sent_pos"],
            examples["sent_score"],
        ],
        axis=1,
    )

    scores = np.array(examples["score"], dtype=float)
    sample_weight = np.log1p(np.maximum(scores, 0.0))

    enc["aux_features"] = aux.tolist()
    enc["labels"] = examples["label"]
    enc["sample_weight"] = sample_weight.tolist()
    return enc


class RobertaWithFeatures(nn.Module):
    def __init__(self, base_name, num_labels, aux_dim=8):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_name, num_labels=num_labels)
        self.roberta = AutoModel.from_pretrained(base_name, config=self.config)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden + aux_dim, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        aux_features=None,
        labels=None,
    ):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS
        pooled = self.dropout(pooled)

        if aux_features is None:
            aux_features = torch.zeros(
                (pooled.size(0), 8), device=pooled.device, dtype=pooled.dtype
            )
        else:
            aux_features = aux_features.to(pooled.dtype)

        fused = torch.cat([pooled, aux_features], dim=-1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        #labels = inputs.get("labels")
        sample_weight = inputs.pop("sample_weight", None)

        outputs = model(**inputs)
        loss = outputs["loss"] 

        if sample_weight is not None:
            sw = sample_weight.to(loss.device).view(-1)
            loss = (loss * sw).mean()
        else:
            loss = loss.mean()

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    from evaluate import load as load_metric
    import numpy as np

    accuracy = load_metric("accuracy")
    f1 = load_metric("f1")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    metrics = {}
    metrics.update(accuracy.compute(predictions=preds, references=labels))
    metrics.update(f1.compute(predictions=preds, references=labels, average="macro"))
    return metrics

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    raw = load_dataset("csv", data_files={"train": CSV_PATH})["train"]

    tokenized = raw.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True,
        remove_columns=raw.column_names,
    )
    tokenized.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "labels",
            "aux_features",
            "sample_weight",
        ],
    )

    splits = tokenized.train_test_split(test_size=0.2, seed=42)
    train_ds = splits["train"]
    val_ds = splits["test"]

    model = RobertaWithFeatures(MODEL_NAME, num_labels=NUM_LABELS, aux_dim=8)

    training_args = TrainingArguments(
        output_dir="models/large_roberta_with_features",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        num_train_epochs=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("models/large_roberta_with_features")
    eval_metrics = trainer.evaluate()
    print(eval_metrics)

if __name__ == "__main__":
    main()
