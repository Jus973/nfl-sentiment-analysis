import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

TRADE_QUALITY = {
    ("Jerry Jeudy", "Broncos", "Browns"): 68,
    ("Jalen Ramsey", "Rams", "Dolphins"): 88,
    ("Tyreek Hill", "Chiefs", "Dolphins"): 96,
    ("DeAndre Hopkins", "Texans", "Cardinals"): 40,
    ("Julio Jones", "Falcons", "Titans"): 55,
    ("AJ Brown", "Titans", "Eagles"): 94,
    ("Amari Cooper", "Cowboys", "Browns"): 78,
    ("Diontae Johnson", "Steelers", "Panthers"): 62,
    ("Davante Adams", "Raiders", "Jets"): 85,
    ("Mac Jones", "Patriots", "Jaguars"): 50,
    ("Joe Mixon", "Bengals", "Texans"): 72,
}

CSV_PATH = "data/processed/reddit/model_labeled_comments.csv"

def get_linear_score(pred):
    if pred == 2: return 1.0
    if pred == 0: return 0.0
    if pred == 1: return -1.0
    return np.nan

def main():
    df = pd.read_csv(CSV_PATH)

    df["trade_key"] = list(zip(df["player_name"], df["old_team"], df["new_team"]))
    
    df = df[df["trade_key"].isin(TRADE_QUALITY.keys())].copy()

    # 0=Neu, 1=Bad, 2=Good, 3=Ignore
    df["linear_sentiment"] = df["student_pred"].apply(get_linear_score)
    df = df.dropna(subset=["linear_sentiment"])

    agg = (
        df.groupby("trade_key")["linear_sentiment"]
        .mean()
        .reset_index()
        .rename(columns={"linear_sentiment": "avg_sentiment"})
    )

    agg["trade_quality"] = agg["trade_key"].map(TRADE_QUALITY)
    
    agg = agg.dropna(subset=["trade_quality", "avg_sentiment"])

    if len(agg) < 2:
        print("Not enough data points to plot.")
        return

    x = agg["trade_quality"].to_numpy()
    y = agg["avg_sentiment"].to_numpy()

    corr, pval = pearsonr(x, y)
    print(f"Pearson r: {corr:.3f}, p-value: {pval:.3g}")
    print(f"Trades plotted: {len(agg)}")

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    scatter = ax.scatter(
        x, y, 
        s=200, 
        alpha=0.7, 
        color="#2E86AB",  
        edgecolors="#1B3A5C", 
        linewidth=1.5,
        zorder=3
    )
    
    m, b = np.polyfit(x, y, 1)
    x_line = np.array([x.min() - 5, x.max() + 5])
    y_line = m * x_line + b
    ax.plot(
        x_line, y_line,
        color="#A23B72",  
        linestyle="-",
        linewidth=2.5,
        alpha=0.8,
        label=f"Linear fit: y = {m:.4f}x + {b:.3f}"
    )
    
    ax.axhline(0, color="#666666", linestyle="--", alpha=0.4, linewidth=1.5, zorder=1)
    
    ax.set_xlabel("Trade Quality Score (1–100)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Sentiment\n(−1 = Negative, 0 = Neutral, +1 = Positive)", 
                   fontsize=13, fontweight="bold")
    ax.set_title(
        f"NFL Trade Quality vs. Reddit User Sentiment\nPearson r = {corr:.3f} (p = {pval:.2e})",
        fontsize=15,
        fontweight="bold",
        pad=20
    )
    
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95, edgecolor="black")
    
    ax.set_xlim(x.min() - 5, 100)
    ax.set_ylim(y.min() - 0.3, y.max() + 0.3)
    
    for i, row in agg.iterrows():
        player = row["trade_key"][0].split()[-1] 
        ax.annotate(
            player,
            xy=(row["trade_quality"], row["avg_sentiment"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="500",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#333333",
                alpha=0.8,
                linewidth=0.5
            ),
            zorder=4
        )
    
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(1.5)
    
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig("trade_quality_vs_sentiment_linear.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
