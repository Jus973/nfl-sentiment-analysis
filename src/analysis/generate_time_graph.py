import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Hardcoded trade list
TRADES = [
    ("Jerry Jeudy", "Broncos", "Browns"),
    ("Jalen Ramsey", "Rams", "Dolphins"),
    ("Tyreek Hill", "Chiefs", "Dolphins"),
    ("DeAndre Hopkins", "Texans", "Cardinals"),
    ("Julio Jones", "Falcons", "Titans"),
    ("AJ Brown", "Titans", "Eagles"),
    ("Amari Cooper", "Cowboys", "Browns"),
    ("Diontae Johnson", "Steelers", "Panthers"),
    ("Davante Adams", "Raiders", "Jets"),
    ("Mac Jones", "Patriots", "Jaguars"),
    ("Joe Mixon", "Bengals", "Texans"),
]

CSV_PATH = "data/processed/reddit/model_labeled_comments.csv"

def get_linear_score(pred):
    if pred == 2: return 1.0  
    if pred == 0: return 0.0 
    if pred == 1: return -1.0 
    return np.nan

def main():
    df = pd.read_csv(CSV_PATH)

    for col in ["player_name", "old_team", "new_team"]:
        df[col] = df[col].astype(str).str.strip()

    df["trade_key"] = list(zip(df["player_name"], df["old_team"], df["new_team"]))
    df = df[df["trade_key"].isin(TRADES)].copy()

    df["linear_sentiment"] = df["student_pred"].apply(get_linear_score)
    df = df.dropna(subset=["linear_sentiment"])

    df["timestamp"] = pd.to_datetime(df["created_utc"])

    plot_data = []

    for trade_key, group in df.groupby("trade_key"):
        group = group.sort_values("timestamp")
        
        start_time = group["timestamp"].min()
        
        group["days_since_start"] = (group["timestamp"] - start_time).dt.total_seconds() / (3600 * 24)
        
        group["day_bin"] = group["days_since_start"].astype(int)
        daily_sentiment = group.groupby("day_bin")["linear_sentiment"].mean()
        rolling_sentiment = daily_sentiment.rolling(window=2, min_periods=1, center=True).mean()
        x_days = rolling_sentiment.index
        
        plot_data.append({
            "label": f"{trade_key[0].split()[-1]} ({trade_key[1]}→{trade_key[2]})",
            "x": x_days,
            "y": rolling_sentiment.values,
            "max_time": 20
        })


    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, len(plot_data)))

    max_x_val = 0
    for i, data in enumerate(plot_data):
        ax.plot(
            data["x"], 
            data["y"], 
            label=data["label"], 
            color=colors[i], 
            linewidth=2.5, 
            alpha=0.8
        )
        if data["max_time"] > max_x_val:
            max_x_val = data["max_time"]

    ax.axhline(0, color="#666666", linestyle="--", alpha=0.5, linewidth=1.5)

    ax.set_xlabel("Days Since First Comment", fontsize=13, fontweight="bold")
    ax.set_ylabel("Rolling Avg Sentiment\n(2-Day Window)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Long-Term Sentiment Evolution Post-Trade\n(Aggregated Daily)",
        fontsize=15,
        fontweight="bold",
        pad=20
    )

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_x_val)
    ax.set_ylim(-1.0, 1.0)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=10)

    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig("sentiment_over_months.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
