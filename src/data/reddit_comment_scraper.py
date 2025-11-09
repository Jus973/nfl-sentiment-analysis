import pandas as pd
import os
import datetime
from src.utils.reddit_utils import init_reddit
import time
import prawcore

def extract_comments(posts_csv, output_csv):
    reddit = init_reddit()
    df_posts = pd.read_csv(posts_csv)
    all_comments = []

    for _, row in df_posts.iterrows():
        post_id = row["id"]
        trade_note = row.get("trade_note", "")
        
        print(row)

        try:
            parts = trade_note.split(" from ")
            player_name = parts[0].strip()
            old_new = parts[1].split(" to ")
            old_team = old_new[0].strip()
            new_team = old_new[1].strip()
        except Exception:
            player_name, old_team, new_team = "", "", ""

        try:
            submission = reddit.submission(id=post_id)

            while True:
                try:
                    submission.comments.replace_more(limit=None)
                    break
                except prawcore.exceptions.TooManyRequests as e:
                    wait = int(e.response.headers.get("Retry-After", 60))
                    print(f"Rate limited. Sleeping for {wait} seconds...")
                    time.sleep(wait)
            
            for comment in submission.comments.list():

                print (comment.id, comment.body)
                all_comments.append({
                    "post_id": post_id,
                    "trade_note": trade_note,
                    "player_name": player_name,
                    "old_team": old_team,
                    "new_team": new_team,
                    "comment_id": comment.id,
                    "author": str(comment.author),
                    "body": comment.body,
                    "score": comment.score,
                    "created_utc": datetime.datetime.fromtimestamp(comment.created_utc)
                })
        
        except Exception as e:
            print(f"Error fetching comments for {post_id}: {e}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(all_comments).to_csv(output_csv, index=False)
    print(f"Saved {len(all_comments)} comments to {output_csv}")

if __name__ == "__main__":
    posts_csv = "data/raw/reddit/scraped_posts.csv"
    output_csv = "data/raw/reddit/scraped_comments.csv"
    extract_comments(posts_csv, output_csv)
