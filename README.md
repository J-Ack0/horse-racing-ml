# Horse Racing Prediction System

An end-to-end ML pipeline for predicting horse racing outcomes and identifying value bets — built from scratch with real daily usage driving every design decision.

The commit history is intentionally unpolished. This was a live system used daily, not a demo project. The cleanup here reflects maturity, not a rewrite.

---

## Model Performance

Trained on ~95,000 historical races with a strict chronological split (no leakage).

| Metric | Value |
|---|---|
| AUC-ROC | 0.737 (95% CI: 0.726–0.748) |
| Average Precision | 0.210 (95% CI: 0.194–0.226) |
| Optimal F1 Threshold | 0.200 |
| Precision at threshold | 21.1% vs 8.8% baseline |
| Improvement over random | **2.4×** |

---

## Project Structure

```
horse-racing-ml/
├── data_collection/
│   ├── scrape_racecards.py     # Scrapes tomorrow's race cards (BS4)
│   ├── scrape_odds.py          # Scrapes live odds + trend analysis (Selenium)
│   └── scrape_results.py       # Scrapes actual results for model auditing
│
├── pipeline/
│   ├── inference.py            # Runs XGBoost predictions on race card input
│   └── betting_filters.py      # Kelly Criterion stake sizing + value bet filter
│
├── ml/
│   └── train.py                # XGBoost training with chronological CV
│
├── models/
│   ├── xgb_model.joblib        # Trained XGBoost classifier
│   ├── xgb_scaler.joblib       # Feature scaler
│   └── label_encoders.pkl      # Categorical encoders
│
├── analysis/
│   ├── EDA.ipynb               # Exploratory analysis
│   ├── EDA-ML.ipynb            # Feature importance + model diagnostics
│   ├── EDA-CAT.ipynb           # Categorical feature analysis
│   └── EDA-NUM.ipynb           # Numerical feature distributions
│
└── dashboard.html              # Local HTML dashboard for daily review
```

---

## Daily Workflow

**1. Scrape tomorrow's race cards**
```bash
python data_collection/scrape_racecards.py
# → Inference_Inputs/Input_YYYY-MM-DD.csv
```

**2. Scrape live odds**
```bash
python data_collection/scrape_odds.py
# → Inference_Odds/Odds_YYYY-MM-DD.csv
# Requires ChromeDriver. Includes shortening/drifting/stable trend signals.
```

**3. Run predictions**
```bash
python pipeline/inference.py
# → Inference_Outputs/Output_YYYY-MM-DD.csv
# → Inference_Logs/YYYY-MM-DD_logs.txt
```

**4. Generate value bets**
```bash
python pipeline/betting_filters.py
# → Inference_Betting/Bets_YYYY-MM-DD.csv
# Applies Kelly Criterion + edge/EV filters
```

**5. Audit results (next day)**
```bash
python data_collection/scrape_results.py
# → Inference_Actuals/YYYY_MM_DD_Actual.csv
# → Inference_Summary/YYYY_MM_DD_Summary_Log.csv
```

**6. Review in dashboard**

Open `dashboard.html` locally — select a date to load predictions, value bets, Lucky 15 combinations, and ROI summary.

---

## Feature Engineering

20 features, all computed with strict point-in-time logic to prevent data leakage.

**Form**
- `EMA_Form` — exponential moving average of recent finishing positions
- `Recent_win_rate` — win rate over last 5 races
- `Career_win_rate` — lifetime win percentage

**Jockey / Trainer**
- `Wilson_score` — confidence-adjusted win rate (handles small sample sizes)
- `jt_runs`, `jt_wins` — historical partnership performance

**Track context**
- `Track_win_rate` — horse's success rate at the specific venue
- `Performance_on_going` — win rate on current ground condition

**Horse attributes**
- Age bins (young / prime / veteran), weight, days since last race, official rating

**Anti-leakage implementation**
```python
# All historical stats use cumulative-up-to-but-not-including-current-race
df.groupby('horse_name')['win'].cumsum() - df['win']  # running total excl. current
df.sort_values('race_date').groupby('horse_name').cumcount()  # chronological count
```

---

## Betting System

### Value identification
- Minimum edge: 1% (model probability − implied probability from odds)
- Minimum EV: 1%
- Strong bet flag: 15%+ edge

### Stake sizing (Quarter Kelly)
```python
kelly = (b * p - q) / b
stake = 0.25 * kelly * bankroll  # Quarter Kelly — conservative, drawdown-aware
```

### Odds trend adjustment
Odds scraped via Selenium are compared across two snapshots:
- Shortening → +2% edge bonus
- Drifting → −2% edge penalty

---

## Model Training

```bash
python ml/train.py
```

**Configuration**
- Algorithm: XGBoost (histogram-based)
- Split: 64% train / 16% val / 20% test (chronological — no shuffle)
- Learning rate: 0.01, max depth: 5, estimators: 780
- Data: ~95,000 races from `merged_v2_horses_labeled.csv`

Threshold selection is based on F1 optimisation at 0.200, accepting lower precision for higher recall — appropriate for a system where identifying more winners matters more than being exact.

---

## Installation

```bash
pip install pandas numpy scikit-learn xgboost beautifulsoup4 requests selenium tqdm joblib matplotlib seaborn
```

ChromeDriver required for odds scraping (must match your Chrome version).

---

## Known Limitations

- Model underperforms on horses with fewer than 2 career runs (insufficient history)
- Racing Post scraper selectors may need updating if site structure changes
- Odds scraping requires JavaScript execution via Selenium — headless Chrome only

---

## Disclaimer

Built for research and learning. Betting carries real financial risk. Past model performance does not predict future results.