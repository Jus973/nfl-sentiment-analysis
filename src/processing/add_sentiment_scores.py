import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

INPUT_CSV = "data/processed/reddit/lex_labeled_with_generic_comments.csv"
OUTPUT_CSV = "data/processed/reddit/full_pre_model_comment_set.csv"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def load_sentiment_model(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def get_sentiment_probs(texts, tokenizer, model, device, batch_size=32):
    all_neg, all_neu, all_pos = [], [], []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = [t if isinstance(t, str) else "" for t in texts[i : i + batch_size]]

            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=160,
            ).to(device)

            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            all_neg.extend(probs[:, 0].tolist())
            all_neu.extend(probs[:, 1].tolist())
            all_pos.extend(probs[:, 2].tolist())

    return all_neg, all_neu, all_pos


def main():
    df = pd.read_csv(INPUT_CSV)

    tokenizer, model, device = load_sentiment_model()

    neg, neu, pos = get_sentiment_probs(
        df["text"].tolist(), tokenizer, model, device
    )

    df["sent_neg"] = neg
    df["sent_neu"] = neu
    df["sent_pos"] = pos

    # composite sentiment score in [-1, 1]:
    # -1 = strongly negative, +1 = strongly positive
    # uses (pos - neg) / (pos + neg + neu)
    denom = df["sent_neg"] + df["sent_neu"] + df["sent_pos"] + 1e-8
    df["sent_score"] = (df["sent_pos"] - df["sent_neg"]) / denom

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved sentiment-augmented CSV to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
