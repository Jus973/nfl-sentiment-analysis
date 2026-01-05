import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import pandas as pd
from src.modeling.roberta_with_features import RobertaWithFeatures
from transformers import AutoTokenizer, AutoConfig

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MAX_LEN = 160

TEACHER_PATH = "models/roberta_with_features"
STUDENT_PATH = "models/student_distilled"
CSV_PATH = "data/processed/reddit/full_comment_set.csv"
TEXT_COLUMN = "text"
NUM_SAMPLES = 2000
BATCH_SIZE = 32

def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def load_teacher(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    config = AutoConfig.from_pretrained(path)
    model = RobertaWithFeatures("roberta-base", 4, aux_dim=8)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def measure_inference_time(model, tokenizer, texts, batch_size=BATCH_SIZE, warmup_batches=5):
    with torch.no_grad():
        for i in range(0, min(len(texts), warmup_batches * batch_size), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            _ = model(**enc)

    start = time.time()
    n = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Benchmarking"):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            _ = model(**enc)
            n += len(batch)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()

    elapsed = time.time() - start
    return elapsed, n / elapsed


def main():
    df = pd.read_csv(CSV_PATH)
    texts = df[TEXT_COLUMN].astype(str).tolist()
    if len(texts) > NUM_SAMPLES:
        texts = texts[:NUM_SAMPLES]
    print(f"Benchmarking on {len(texts)} samples on {DEVICE}")

    teacher, teacher_tok = load_teacher(TEACHER_PATH)
    student, student_tok = load_model(STUDENT_PATH)

    t_time, t_throughput = measure_inference_time(
        teacher, teacher_tok, texts, batch_size=BATCH_SIZE
    )
    s_time, s_throughput = measure_inference_time(
        student, student_tok, texts, batch_size=BATCH_SIZE
    )

    print("\n=== Inference speed ===")
    print(f"Teacher total time: {t_time:.2f}s, throughput: {t_throughput:.2f} samples/s")
    print(f"Student total time: {s_time:.2f}s, throughput: {s_throughput:.2f} samples/s")
    print(f"Speedup (teacher_time / student_time): {t_time / s_time:.2f}x")

if __name__ == "__main__":
    main()
