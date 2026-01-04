import torch
import os
import sys
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from src.modeling.roberta_with_features import RobertaWithFeatures 
import safetensors.torch as st

TEACHER_PATH = "models/roberta_with_features"
BASE_MODEL = "roberta-base"
INPUT_CSV = "data/processed/reddit/full_pre_model_comment_set.csv"
OUT_NPY = "data/processed/reddit/teacher_logits.npy"
MAX_LENGTH = 160
NUM_LABELS = 4
AUX_DIM = 8 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    teacher = RobertaWithFeatures(BASE_MODEL, num_labels=NUM_LABELS, aux_dim=AUX_DIM)
    state_dict = st.load_file(f"{TEACHER_PATH}/model.safetensors")
    teacher.load_state_dict(state_dict)
    teacher.eval().to(device)

    dataset = load_dataset("csv", data_files={"train": INPUT_CSV})["train"]

    all_logits = []
    batch_size = 32

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        texts = batch["text"]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        aux = torch.tensor(
            list(
                zip(
                    batch["lex_pos"],
                    batch["lex_neu"],
                    batch["lex_neg"],
                    batch["lex_relevance"],
                    batch["sent_neg"],
                    batch["sent_neu"],
                    batch["sent_pos"],
                    batch["sent_score"],
                )
            ),
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            outputs = teacher(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                aux_features=aux,
            )
        all_logits.append(outputs["logits"].cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    np.save(OUT_NPY, logits)
    print(f"Saved logits for {logits.shape[0]} examples to {OUT_NPY}")


if __name__ == "__main__":
    main()
