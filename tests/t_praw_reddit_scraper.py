import praw
import json

with open("config/reddit_config.json") as f:
    creds = json.load(f)

reddit = praw.Reddit(
    client_id=creds["client_id"],
    client_secret=creds["client_secret"],
    user_agent=creds["user_agent"]
)

reddit.read_only=True

subreddit = reddit.subreddit("nfl")
for post in subreddit.hot(limit=5):
    print(f"{post.title} (Score: {post.score})")