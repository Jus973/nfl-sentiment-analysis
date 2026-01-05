# NFL Trade Sentiment Analysis: Predicting Trade Outcomes from Reddit Discourse

## Objective

Determine whether sentiment from Reddit discussions accurately predicts the long-term success of NFL trades. 
**Can fans accurately predict how an NFL trade will go?**

---

## Technical Approach

### 1. Data Collection
- Scraped **~25k+ Reddit comments** from Reddit trade discussion threads using the PRAW API.
- Captured metadata: author, comment text, upvote score, timestamp, and associated trade (player, teams).
- Filtered to posts within 40 days of official trade announcements for relevance.

### 2. Data Processing & Labeling
- **Hand-labeled ~1.2k comments** across sentiment (positive/negative/neutral) and relevance (trade-discussion vs. off-topic).
- Balanced labeling to ensure robust model training across all sentiment classes.
- Added **auxilliary input features** including enriched text (Player + Team Context + Comment) and **lexical features** (VADER scores, keyword density).

### 3. Teacher Model: Sentiment & Relevance Classifier
- **Fine-tuned RoBERTa-base** on manually labeled data as a unified **4-class classifier** (Negative, Neutral, Positive, Irrelevant).
- Used a 80% training, 20% validation data split
- Architecture: ~355M parameters, **achieved competitive Macro-F1** on the validation set but is computationally expensive for large-scale inference.
- Auxilliary input features led to a **31% increase in model accuracy** and **40% increase in macro-F1.**

### 5. Knowledge Distillation: Student Model
- Distilled the teacher into **DistilBERT-base** using a blended knowledge distillation loss:
  `L = α · L_KD + (1 − α) · L_CE`, with `α = 0.7` and temperature `T = 2.0`.
- **Student model:** ~66M parameters (~5.4× smaller, ~6× faster inference).
- Distilled model retains teacher's soft probabilistic knowledge while reducing computational footprint.

### 6. Trade-Level Aggregation & Outcome Evaluation
- Aggregated student predictions to compute **average sentiment per trade**.
- Mapped discrete predictions to linear scale:
  - **2 (Good)** → +1.0
  - **1 (Bad)** → −1.0
  - **0 (Neutral)** → 0.0
  - **3 (Irrelevant)** → Dropped
- Correlated average sentiment against expert-assigned trade quality scores (1–100 scale).

---

## Key Findings

![Trade Quality vs. Reddit Sentiment](trade_quality_vs_sentiment_linear.png)

**Correlation Analysis:**
- **Pearson r = −0.52** (p < 0.05)
- Trades rated as high-quality tend to generate *more negative* Reddit sentiment, and vice versa.

**Interpretation:**
- Analysis based on 10 trades with sufficient comment volume and sentiment signal
- A moderate negative correlation (**r = −0.52**) indicates that real-world outcomes often move **opposite** to initial fan sentiment, suggesting that early reactions are frequently misaligned with eventual results.

---

<p align="center">
  <img src="sentiment_over_months.png" alt="Sentiments Over Months" width="45%">
  <img src="trade_sentiment_stacked_bar.png" alt="Trade Sentiment Stacked Bar" width="45%">
</p>

## Results

- Aggregated Reddit sentiment exhibits a **moderate inverse correlation** with expert trade evaluations (**Pearson r = −0.52, p < 0.05**), suggesting that fan predictions are often the opposite of long-term results
- The distilled student model retains **~96% of the teacher’s macro-F1** while significantly reducing model complexity.
- Knowledge distillation yields a **~1.98× faster inference time**, enabling scalable trade-level sentiment analysis.

---

## Repository Structure

| Directory | Purpose |
|-----------|---------|
| `data/processed/reddit/` | Cleaned, labeled, and model-predicted comment datasets |
| `data/raw/` | Raw Reddit scrapes (original PRAW output) |
| `models/` | Pre-trained teacher, distilled student, and benchmark logs |
| `processing/` | Data pipeline scripts (merge, label, clean) |
| `modeling/` | Model training, distillation, and inference code |
| `analysis/` | Downstream analysis scripts and visualizations |

## Performance & Efficiency

| Metric                     | Teacher (RoBERTa-large) | Student (DistilBERT-base) | Improvement |
|----------------------------|-------------------------|----------------------------|-------------|
| Model Size                 | ~355M parameters        | ~66M parameters            | **5.4× smaller** |
| Inference Throughput (GPU) | Baseline                | ~1.98× faster              | **1.98× speedup** |
| Macro F1 Retention         | 100%                    | ~96%                       | **−4% absolute** |
| Distillation Loss          | N/A                     | 0.71 (KD + CE blended)     | — |

## Limitations
- The analysis focuses on sentiment surrounding 11 NFL trades, which may not fully capture broader league-wide or season-long sentiment trends.
- Only 1.2k comments were manually labeled due to time constraints. Expanding the labeled dataset would likely improve model generalization and robustness.
- Human labeling introduced an uneven class distribution, particularly an overrepresentation of the *Irrelevant* class, which may affect minority-class performance despite mitigation efforts.

## Future Directions
- Incorporate more NFL seasons and trades to increase sample size, and manually label more comments to increase model accuracy
- Separate 4-class model into two separate models, with an irrelevance filter first and then a 3 class neutral/good/bad sentiment model in hopes of simplifying model task for higher accuracy. 

## References
- Hinton, G., Vanhoucke, V., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network." arXiv:1503.02531.
- Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." ICLR.
- Sanh, V., Debut, L., et al. (2019). "DistilBERT, a distilled version of BERT." arXiv:1910.01108.
