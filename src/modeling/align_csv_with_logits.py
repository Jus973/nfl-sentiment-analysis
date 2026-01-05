import pandas as pd
import numpy as np

csv_path = "data/processed/reddit/full_pre_model_comment_set.csv"
logits_path = "data/processed/reddit/teacher_logits.npy"

df = pd.read_csv(csv_path)
teacher_logits = np.load(logits_path)
assert len(df) == teacher_logits.shape[0]

df["_idx"] = np.arange(len(df))
tmp_csv = "data/processed/reddit/_tmp_distill.csv"
df.to_csv(tmp_csv, index=False)
