import json
import praw
from urllib.parse import urlparse

def load_config(path="config/reddit_config.json"):
    with open(path) as f:
        return json.load(f)

def init_reddit(config=None):
    if config is None:
        config = load_config()
    reddit = praw.Reddit(
        client_id=config["client_id"],
        client_secret=config["client_secret"],
        user_agent=config["user_agent"]
    )
    reddit.read_only = True
    return reddit

def extract_id_from_url(url):
    path_parts = urlparse(url).path.split("/")
    if "comments" in path_parts:
        return path_parts[path_parts.index("comments") + 1]
    return None
