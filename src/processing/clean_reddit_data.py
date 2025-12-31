import pandas as pd
import re
import os

SCRAPED_COMMENTS = "data/raw/reddit/scraped_comments.csv"
OUTPUT_FILE = "data/processed/reddit/cleaned_comments.csv"

def clean_comment_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'&gt;.*', '', text)
    text = re.sub(r'\s+', ' ', text)  
    text = text.strip()
    return text

def clean_comments():
    df = pd.read_csv(SCRAPED_COMMENTS)
    cleaned = []

    for _, row in df.iterrows():
        player = str(row.get("player_name", "")).strip()
        old_team = str(row.get("old_team", "")).strip()
        new_team = str(row.get("new_team", "")).strip()
        text = str(row.get("body", "")).strip()

        if not text:
            continue
        
        #TODO make keywords more lenient
        keywords = [player, old_team, new_team]
        if not any(k.lower() in text.lower() for k in keywords if k):
            continue

        cleaned_text = clean_comment_text(text)
        if len(cleaned_text.split()) < 3:
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
