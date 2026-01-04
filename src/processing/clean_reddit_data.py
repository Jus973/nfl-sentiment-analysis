import pandas as pd
import re
import os

SCRAPED_COMMENTS = "data/raw/reddit/scraped_comments.csv"
OUTPUT_FILE = "data/processed/reddit/cleaned_comments.csv"

POSITIVE_CUES = {
    "great", "good", "huge", "steal", "love", "solid", "perfect",
    "win", "upgrade", "elite", "underrated", "value"
}

NEGATIVE_CUES = {
    "bad", "awful", "terrible", "overpaid", "trash", "hate",
    "washed", "regret", "disaster", "loss", "downgrade"
}

NEUTRAL_CUES = {
    "depends", "meh", "fine", "okay", "average", "we'll see"
}

def clean_comment_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'&gt;.*', '', text)
    text = re.sub(r'\s+', ' ', text)  
    text = text.strip()
    return text

def relevance_score(text, player, old_team, new_team):
    text_l = text.lower()
    score = 0

    for entity in [player, old_team, new_team]:
        if entity and entity.lower() in text_l:
            score += 2

    score += sum(1 for w in POSITIVE_CUES if w in text_l)
    score += sum(1 for w in NEGATIVE_CUES if w in text_l)
    score += sum(0.5 for w in NEUTRAL_CUES if w in text_l)

    return score

def clean_comments(min_relevance=1.5, min_words=4):
    df = pd.read_csv(SCRAPED_COMMENTS)
    cleaned = []

    for _, row in df.iterrows():
        player = str(row.get("player_name", "")).strip()
        old_team = str(row.get("old_team", "")).strip()
        new_team = str(row.get("new_team", "")).strip()
        text = str(row.get("body", "")).strip()
        
        if not text:
            continue
        
        cleaned_text = clean_comment_text(text)
        if len(cleaned_text.split()) < min_words:
            continue

        score = relevance_score(cleaned_text, player, old_team, new_team)

        if score < min_relevance:
            continue

        cleaned.append({
            "comment_id": row["comment_id"],
            "player_name": player,
            "old_team": old_team,
            "new_team": new_team,
            "author": row.get("author", ""),
            "text": cleaned_text,
            "score": row.get("score", 0),
            "created_utc": row.get("created_utc", ""),
            "post_id": row.get("post_id", ""),
        })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    pd.DataFrame(cleaned).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(cleaned)} cleaned comments to {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_comments()
