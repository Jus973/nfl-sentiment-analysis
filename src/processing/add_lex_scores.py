import re
import pandas as pd

INPUT_CSV = "data/processed/reddit/labeled_with_generic_comments.csv"
OUTPUT_CSV = "data/processed/reddit/lex_labeled_with_generic_comments.csv"

POSITIVE_CUES = {
    "great", "good", "huge", "steal", "love", "solid", "perfect",
    "win", "upgrade", "elite", "underrated", "value", "robbery",
    "fleeced", "cook", "cooking", "balling", "insane", "nice"
}

NEGATIVE_CUES = {
    "bad", "awful", "terrible", "overpaid", "trash", "hate",
    "washed", "regret", "disaster", "loss", "downgrade", "mid",
    "sell", "fraud", "bust", "horrible", "dogshit", "ass"
}

NEUTRAL_CUES = {
    "depends", "meh", "fine", "okay", "average", "we'll see",
    "idk", "maybe", "could be", "not sure"
}


def clean_comment_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"&gt;.*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_lexicon_hits(tokens):
    pos = neg = neu = 0

    for t in tokens:
        if t in POSITIVE_CUES:
            pos += 1
        if t in NEGATIVE_CUES:
            neg += 1
        if t in NEUTRAL_CUES:
            neu += 1

    return pos, neu, neg


def compute_relevance(text, player, old_team, new_team, pos, neu, neg):
    text_l = text.lower()
    score = 0.0

    for entity in [player, old_team, new_team]:
        if isinstance(entity, str) and entity:
            if entity.lower() in text_l:
                score += 2.0

    score += pos * 1.0
    score += neg * 1.0
    score += neu * 0.5

    # very short comments with no cues are low relevance
    if len(text_l.split()) < 5 and (pos + neu + neg == 0):
        score -= 1.0

    return max(score, 0.0)


def process_row(row):
    raw_text = row.get("text", "")
    text = clean_comment_text(raw_text)
    tokens = re.findall(r"[a-zA-Z']+", text.lower())

    pos, neu, neg = count_lexicon_hits(tokens)

    lex_rel = compute_relevance(
        text=text,
        player=row.get("player_name", ""),
        old_team=row.get("old_team", ""),
        new_team=row.get("new_team", ""),
        pos=pos,
        neu=neu,
        neg=neg,
    )

    row["lex_pos"] = pos
    row["lex_neu"] = neu
    row["lex_neg"] = neg
    row["lex_relevance"] = lex_rel
    return row


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df.apply(process_row, axis=1)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved with lexicon features to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
