import os
import pandas as pd
import datetime
from urllib.parse import urlparse
from src.utils.reddit_utils import init_reddit, extract_id_from_url

def chunk_date_ranges(start_date, end_date, chunk_days=5):
    chunks = []
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + pd.Timedelta(days=chunk_days), end_date)
        chunks.append((current_start, current_end))
        current_start = current_end
    return chunks

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
            "url": f"https://www.reddit.com{submission.permalink}"
        })
    
    print(f"Query: {query}, found {len(posts)} posts from {start_date.date()} to {end_date.date()}")
    return pd.DataFrame(posts)

def filter_posts(posts_df, trade_name):
    player_name = trade_name.replace(" trade", "").strip()
    keywords = ["trade", "traded", "deal", "acquire", "sign", "move"]
    pattern = f"{player_name}|{'|'.join(keywords)}"
    mask = posts_df['title'].str.contains(pattern, case=False, na=False, regex=True)
    return posts_df[mask]


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
        trade_name = row["trade_name"]
        trade_date_str = row["trade_date"]
        trade_date = datetime.datetime.fromisoformat(trade_date_str).replace(tzinfo=datetime.timezone.utc)
        
        start = trade_date - pd.Timedelta(days=20)
        end = trade_date + pd.Timedelta(days=20)
        date_chunks = chunk_date_ranges(start, end, chunk_days=5)

        posts_chunks = []
        for start_chunk, end_chunk in date_chunks:
            chunk_posts = scrape_posts(query, limit=None, start_date=start_chunk, end_date=end_chunk)
            if not chunk_posts.empty:
                posts_chunks.append(chunk_posts)

        if posts_chunks:
            posts_df = pd.concat(posts_chunks, ignore_index=True)
            filtered_posts = filter_posts(posts_df, trade_name)
            if not filtered_posts.empty:
                filtered_posts = filtered_posts.copy()
                filtered_posts["trade_note"] = trade_name
                all_posts.append(filtered_posts)

    manual_df = pd.read_csv("data/inputs/trade_posts.csv")
    manual_df["id"] = manual_df["url"].apply(extract_id_from_url)
    manual_df = manual_df.dropna(subset=["id"])
    manual_df["trade_note"] = manual_df["trade_name"]

    combined_df = pd.concat(
        [
            pd.concat(all_posts, ignore_index=True) if all_posts else pd.DataFrame(columns=["id", "title", "url", "trade_note"]),
            manual_df[["id", "url", "trade_note"]]
        ],
        ignore_index=True
    )

    ids_df = combined_df[["id", "trade_note", "url"]].drop_duplicates()
    output_file = os.path.join(reddit_folder, "scraped_posts.csv")
    ids_df.to_csv(output_file, index=False)
    print(f"Saved {len(ids_df)} posts to {output_file}")
