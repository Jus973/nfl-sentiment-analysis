import pandas as pd
df = pd.read_csv("data/inputs/trade_queries.csv")
print(df["query"].tolist())
