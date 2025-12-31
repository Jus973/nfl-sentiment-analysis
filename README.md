# NFL Sentiment Analysis

Objective
-

- Analyze public and expert sentiment of NFL trades and evaluate how accurate this sentiment predicts the long-term results of these actual trades
- Leverage Reddit, X, and expert media sources as inputs


Components
- 

- Data Collection
- Data Processing
  - Manually label half of the original scraped comments, then have BertE label the rest
- Model Distillation
- Sentiment Analysis
  - Account for sarcasm i.e.
    - comment_id: i1tkk80
      player: Tyreek Hill   Chiefs -> Dolphins
      score: 2   author: guessmyGTRaintShit
      text:
      Now the dolphins will finish 6-11 instead of 5-12! What a deal!
- Outcome Evaluation
- Predictive Modeling


Script Order:

- Fill in inputs/trade_queries.csv for trades to analyze
- data/reddit_post_scraper.py
- Optionally add inputs/trade_posts.csv for hand-selected posts with good data
- data/reddit_comment_scraper.py
- processing/clean_reddit_data.py
- processing/manual_label_comments.py
- processing/BERT_label_comments.py
- modeling/fine_tune_sentiment