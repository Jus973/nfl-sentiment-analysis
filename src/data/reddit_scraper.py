import os
import pandas as pd
import datetime
import praw
import json
from urllib.parse import urlparse

def load_config():
    with open("config/reddit_config.json") as f:
        return json.load(f)

def init_reddit():
    config=load_config()
    reddit = praw.Reddit(
        client_id=config["client_id"],
        client_secret=config["client_secret"],
        user_agent=config["user_agent"]
    )
    reddit.read_only=True
    return reddit

def scrape_posts(query, subreddit="nfl", limit=100, sort="new", start_date=None, end_date=None):
    reddit = init_reddit()
    posts = []

    for submission in reddit.subreddit(subreddit).search(query, limit=limit, sort=sort):
        post_time = datetime.datetime.fromtimestamp(submission.created_utc, tz=datetime.timezone.utc)
        if start_date and post_time < start_date:
            continue
        if end_date and post_time > end_date:
            continue

        posts.append({
            "id": submission.id,
            "title": submission.title,
            "url": submission.url
        })
    
    print(f"Query: {query}, found {len(posts)} posts")
    return pd.DataFrame(posts)


def extract_id_from_url(url):
    path_parts = urlparse(url).path.split("/")
    if "comments" in path_parts:
        return path_parts[path_parts.index("comments") + 1]
    return None

if __name__ == "__main__":
    reddit_folder = "data/raw/reddit"
    os.makedirs(reddit_folder, exist_ok=True)

    queries_df = pd.read_csv("data/inputs/trade_queries.csv") 
    all_posts = []

    for _, row in queries_df.iterrows():
        query = row["query"]
        trade_note = row.get("note", query)
        
        trade_date_str = row.get("trade_date")
        trade_date = datetime.datetime.fromisoformat(trade_date_str).replace(tzinfo=datetime.timezone.utc)
        start = trade_date - pd.Timedelta(days=20)
        end = trade_date + pd.Timedelta(days=20)
        
        posts_df = scrape_posts(query, limit=50000, start_date=start, end_date=end)
        if not posts_df.empty:
            posts_df["trade_note"] = trade_note
            all_posts.append(posts_df)

    manual_df = pd.read_csv("data/inputs/trade_posts.csv")
    manual_df["id"] = manual_df["url"].apply(extract_id_from_url)
    manual_df = manual_df.dropna(subset=["id"])
    manual_df["trade_note"] = manual_df["trade_name"]

    combined_df = pd.concat([pd.concat(all_posts, ignore_index=True) if all_posts else pd.DataFrame(columns=["id", "title", "url", "trade_note"]),
                             manual_df[["id", "url", "trade_note"]]], ignore_index=True)
    
    ids_df = combined_df[["id", "trade_note", "url"]].drop_duplicates()
    output_file = os.path.join(reddit_folder, "scraped_posts.csv")
    ids_df.to_csv(output_file, index=False)
    print(f"Saved {len(ids_df)} posts to {output_file}")
