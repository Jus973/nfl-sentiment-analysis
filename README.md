# NFL Sentiment Analysis

Objective
-

- Analyze public and expert sentiment of NFL trades and evaluate how accurate this sentiment predicts the long-term results of these actual trades
- Leverage Reddit API as inputs
- Filter irrelevant conversations


Components
- 

- Data Collection
- Data Processing
  - Manually label half of the original scraped comments
- Train irrelevance filter model
  - Reddit conversations quickly turn unproductive in pure trade analysis sentiment
- Train teacher model -> fine-tune BERT model
  - Set 70% of labeled data to train, 15% for validation, and 15% for testing
- Distill to a student model with DistillBERT with distillation loss and plus cross-entropy
- Use the student for trade-level analysis
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