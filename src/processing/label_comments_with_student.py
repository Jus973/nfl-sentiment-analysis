import time
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

INPUT_CSV = "data/processed/reddit/full_comment_set.csv"
OUTPUT_CSV = "data/processed/reddit/model_labeled_comments.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
ONLY_UNLABELED = True 

STUDENT_PATH = "models/student_distilled"
MODEL_NAME = "distilbert-base-uncased" 
MAX_LEN = 160
BATCH_SIZE = 64

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT_PATH)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def batched_predict(model, tokenizer, texts, batch_size=BATCH_SIZE):
    all_logits = []
    all_preds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Labeling"):
            batch_texts = texts[i : i + batch_size]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            outputs = model(**enc)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_logits.append(logits.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_preds, axis=0),
    )

def main():
    df = pd.read_csv(INPUT_CSV)

    if ONLY_UNLABELED and LABEL_COLUMN in df.columns:
        mask = df[LABEL_COLUMN].isna()
    else:
        mask = np.ones(len(df), dtype=bool)

    texts = df.loc[mask, TEXT_COLUMN].astype(str).tolist()
    print(f"Running student on {len(texts)} rows (out of {len(df)})")

    model, tokenizer = load_model_and_tokenizer()

    start = time.time()
    logits, preds = batched_predict(model, tokenizer, texts)
    elapsed = time.time() - start

    print(
        f"Inference time: {elapsed:.2f} s, "
        f"{len(texts) / elapsed:.2f} examples/s"
    )

    df.loc[mask, "student_pred"] = preds

    if LABEL_COLUMN in df.columns:
        df.loc[df[LABEL_COLUMN].notna(), "student_pred"] = df.loc[df[LABEL_COLUMN].notna(), LABEL_COLUMN]
    
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved labeled CSV to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
