import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

TEACHER_PATH = "models/large_teacher_4class_class_weighted"
MODEL_NAME = "roberta-large"
INPUT_CSV = "data/processed/reddit/labeled_with_generic_comments.csv"
OUT_NPY = "data/processed/reddit/teacher_logits.npy"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_PATH,
        num_labels=4,
    ).eval().to(device)

    dataset = load_dataset("csv", data_files={"train": INPUT_CSV})["train"]

    all_logits = []
    batch_size = 32

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        texts = batch["text"]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=160,
        ).to(device)

        with torch.no_grad():
            outputs = teacher(**inputs)
        all_logits.append(outputs.logits.cpu().numpy())
    
    logits = np.concatenate(all_logits, axis=0)
    np.save(OUT_NPY, logits)
    print(f"Saved logits for {logits.shape[0]} examples to {OUT_NPY}")
