import pandas as pd
import re
import os

SCRAPED_COMMENTS = "data/raw/reddit/scraped_comments.csv"
TRADE_QUERIES = "data/inputs/trade_queries.csv"
OUTPUT_FILE = "data/processed/cleaned_comments.csv"

def parse_trade_info(trade_query):
    player_match = re.search(r'"(.+?)"', trade_query)
    player = player_match.group(1) if player_match else None

    to_team_match = re.search(r'trade to (.+)', trade_query, re.IGNORECASE)
    to_team = to_team_match.group(1).strip() if to_team_match else None

    from_team_match = re.search(r'trade from (.+?) to', trade_query, re.IGNORECASE)
    from_team = from_team_match.group(1).strip() if from_team_match else None

    return player, from_team, to_team

def clean_comment_text(text):
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'\s+', ' ', text) 
    text = text.strip()
    return text

comments_df = pd.read_csv(SCRAPED_COMMENTS)
trades_df = pd.read_csv(TRADE_QUERIES)

trade_info_dict = {}
for _, row in trades_df.iterrows():
    trade_note = row["trade_name"]
    player, from_team, to_team = parse_trade_info(row["query"])
    trade_info_dict[trade_note] = {
        "player": player,
        "from_team": from_team,
        "to_team": to_team
    }

cleaned_comments = []
for _, row in comments_df.iterrows():
    trade_note = row["trade_note"]
    if trade_note not in trade_info_dict:
        continue

    info = trade_info_dict[trade_note]
    player = info["player"]
    from_team = info["from_team"]
    to_team = info["to_team"]

    text = str(row["body"])
    # Filter for relevance: contains player or teams
    keywords = [kw for kw in [player, from_team, to_team] if kw]
    if not any(kw.lower() in text.lower() for kw in keywords):
        continue

    cleaned_text = clean_comment_text(text)
    if len(cleaned_text.split()) < 3:  # skip very short comments
        continue

    cleaned_comments.append({
        "comment_id": row["comment_id"],
        "trade_note": trade_note,
        "player": player,
        "from_team": from_team,
        "to_team": to_team,
        "author": row.get("author", ""),
        "text": cleaned_text,
        "score": row.get("score", 0),
        "created_utc": row.get("created_utc", "")
    })

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
pd.DataFrame(cleaned_comments).to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(cleaned_comments)} cleaned comments to {OUTPUT_FILE}")
