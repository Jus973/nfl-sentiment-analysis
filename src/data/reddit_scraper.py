import os
import pandas as pd
import datetime
import praw
import json

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
    reddit=init_reddit()
    posts=[]

    for submission in reddit.subreddit(subreddit).search(query, limit=limit, sort=sort):
        post_time = datetime.datetime.fromtimestamp(submission.created_utc, tz=datetime.timezone.utc)
        if start_date and post_time < start_date:
            continue
        if end_date and post_time > end_date:
            continue
        
        posts.append({
            "id": submission.id,
            "title": submission.title,
            "selftext": submission.selftext,
            "score": submission.score,
            "num_comments": submission.num_comments,
            "created_utc": datetime.datetime.fromtimestamp(submission.created_utc, tz=datetime.timezone.utc),
            "url": submission.url
        })
    
    return pd.DataFrame(posts)

def scrape_comments(submission_id, limit=1000):
    reddit = init_reddit()
    submission = reddit.submission(id=submission_id)
    submission.comments.replace_more(limit=0)
    comments = []
    for comment in submission.comments.list()[:limit]:
        comments.append({
            "comment_id": comment.id,
            "body": comment.body,
            "score": comment.score,
            "created_utc": datetime.datetime.fromtimestamp(comment.created_utc, datetime.timezone.utc)
        })
    return pd.DataFrame(comments)

def save_posts(df, query, folder="data/raw/reddit_posts"):
    os.makedirs(folder, exist_ok=True)
    query_clean = query.replace(" ", "_")
    filename = f"{folder}/{query_clean}_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} posts to {filename}")
    return filename

if __name__ == "__main__":
    query = '"Jerry Jeudy" trade'
    
    trade_date = datetime.datetime(2024, 3, 13, tzinfo=datetime.timezone.utc)
    start = trade_date - pd.Timedelta(days=2)
    end = trade_date + pd.Timedelta(days=2)

    posts_df = scrape_posts(query, limit=50000, start_date=start, end_date=end)
    save_posts(posts_df, query)
