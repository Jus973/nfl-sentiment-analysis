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

    df["trade_label"] = df.apply(
        lambda row: f"{row['player_name'].split()[-1]}\n{row['old_team']}→{row['new_team']}", 
        axis=1
    )

    sentiment_counts = df.groupby(["trade_label", "linear_sentiment"]).size().unstack(fill_value=0)
    
    for sentiment in [-1.0, 0.0, 1.0]:
        if sentiment not in sentiment_counts.columns:
            sentiment_counts[sentiment] = 0

    sentiment_counts = sentiment_counts[[-1.0, 0.0, 1.0]]

    sentiment_pct = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {
        -1.0: "#FF6B6B",  
        0.0: "#FFD93D", 
        1.0: "#6BCB77",
    }

    sentiment_pct.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[colors[s] for s in sentiment_pct.columns],
        edgecolor="#333333",
        linewidth=1.5,
        alpha=0.85,
    )

    ax.set_xlabel("Trade", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sentiment Distribution (%)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Sentiment Composition by Trade\nStacked Percentage Distribution",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    legend_labels = {-1.0: "Negative", 0.0: "Neutral", 1.0: "Positive"}
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors[s], ec="#333333", linewidth=1.5, alpha=0.85)
        for s in [-1.0, 0.0, 1.0]
    ]
    ax.legend(
        handles,
        [legend_labels[s] for s in [-1.0, 0.0, 1.0]],
        loc="upper right",
        fontsize=11,
        framealpha=0.95,
        edgecolor="black",
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5, axis="y")
    ax.set_axisbelow(True)

    ax.set_ylim(0, 100)

    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(1.5)

    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig("trade_sentiment_stacked_bar.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n=== Sentiment Distribution (%) ===")
    print(sentiment_pct.round(1))
    
    print("\n=== Raw Comment Counts ===")
    print(sentiment_counts)

if __name__ == "__main__":
    main()
