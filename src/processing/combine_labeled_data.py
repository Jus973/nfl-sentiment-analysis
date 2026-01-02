import pandas as pd
import os

LABELED_CSV = "data/processed/reddit/labeled_comments.csv"
GENERIC_CSV = "data/inputs/generic_training_comments.csv"
OUTPUT_CSV = "data/processed/reddit/labeled_with_generic_comments.csv"

if __name__ == "__main__":
    labeled_df = pd.read_csv(LABELED_CSV)
    generic_df = pd.read_csv(GENERIC_CSV)

    generic_df = generic_df[labeled_df.columns]
    combined_df = pd.concat([labeled_df, generic_df], ignore_index=True)

    combined_df.to_csv(OUTPUT_CSV, index=False)