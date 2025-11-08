#seems PSAW is deprecated for public and is only for reddit moderators now

from psaw import PushshiftAPI
import datetime

api = PushshiftAPI()

trade_date = datetime.datetime(2019, 10, 15)  
start_epoch = int((trade_date - datetime.timedelta(days=2)).timestamp()) 
end_epoch   = int((trade_date + datetime.timedelta(days=2)).timestamp())

posts = api.search_submissions(
    q="Jalen Ramsey trade",
    subreddit="nfl",
    after=start_epoch,
    before=end_epoch,
    filter=['id', 'title', 'selftext', 'score', 'num_comments', 'created_utc'],
    limit=200
)

for post in posts:
    print(post.title, datetime.datetime.fromtimestamp(post.created_utc))