#create a csv of all comments in cleaned_comments plus those manually labeled
import pandas as pd

cleaned_csv = "data/processed/reddit/cleaned_comments.csv"
labeled_csv = "data/processed/reddit/labeled_comments.csv"
output_csv = "data/processed/reddit/cleaned_with_labels.csv"
id_column = "comment_id"

def main():
    df_clean = pd.read_csv(cleaned_csv)
    df_labeled = pd.read_csv(labeled_csv)

    missing = set(df_labeled[id_column]) - set(df_clean[id_column])
    if missing:
        print(f"Warning: {len(missing)} labeled IDs not found in cleaned CSV")

    df_labels = df_labeled[[id_column, "label"]]

    merged = df_clean.merge(
        df_labels,
        how="left",      
        on=id_column,   
    )

    merged.to_csv(output_csv, index=False)
    print(f"Saved merged CSV with labels to {output_csv}")
    print(f"Total rows: {len(merged)}, labeled rows: {merged['label'].notna().sum()}")


if __name__ == "__main__":
    main()
