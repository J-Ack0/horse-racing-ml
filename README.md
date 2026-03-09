# Horse Racing Prediction & Betting System

A comprehensive machine learning pipeline for predicting horse racing outcomes and identifying value betting opportunities using XGBoost and Kelly Criterion-based stake sizing.

## 📊 System Overview

This system consists of four main components:
1. **Data Collection** - Automated web scraping of race cards and odds
2. **Prediction Engine** - XGBoost model with engineered features
3. **Betting Analysis** - Value bet identification using edge and EV calculations
4. **Performance Tracking** - Automated outcome verification and ROI reporting

## 🎯 Model Performance

**Test Set Metrics (Chronological Split):**
- **AUC-ROC**: 0.737 (95% CI: 0.726-0.748)
- **Average Precision**: 0.210 (95% CI: 0.194-0.226)
- **Optimal F1 Threshold**: 0.200 (Precision: 0.211, Recall: 0.367)

**Strike Rate Analysis:**
- Baseline Random Precision: 8.8%
- Model achieves 2.4x improvement over random at optimal threshold

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost beautifulsoup4 requests selenium tqdm joblib matplotlib seaborn
```

### Daily Workflow

**1. Scrape Tomorrow's Race Cards**
```bash
python bs4-data-test-selector.py
```
- Generates: `Inference_Inputs/Input_YYYY-MM-DD.csv`

**2. Scrape Live Odds**
```bash
python GET_odds.py
```
- Requires Chrome WebDriver
- Generates: `Inference_Odds/Odds_YYYY-MM-DD.csv`
- Includes trend analysis (shortening/drifting/stable)

**3. Generate Predictions**
```bash
python early_pred_2026-01.py
```
- Generates: `Inference_Outputs/Output_YYYY-MM-DD.csv`
- Includes detailed logs: `Inference_Logs/YYYY-MM-DD_logs.txt`

**4. Identify Value Bets**
```bash
python VOID___betting_filters.py
```
- Generates: `Inference_Betting/Bets_YYYY-MM-DD.csv`
- Applies Kelly Criterion for stake sizing

**5. Audit Results (Next Day)**
```bash
python Actual_Outcome_Inference_Scraper.py
```
- Generates: 
  - `Inference_Actuals/YYYY_MM_DD_Actual.csv`
  - `Inference_Summary/YYYY_MM_DD_Summary_Log.csv`
- Tracks betting ROI and prediction accuracy

## 📁 Project Structure

```
project/
│
├── bs4-data-test-selector.py      # Scrapes race cards
├── GET_odds.py                     # Scrapes odds with Selenium
├── early_pred_2026-01.py          # Makes predictions
├── VOID___betting_filters.py      # Analyzes betting value
├── Actual_Outcome_Inference_Scraper.py  # Audits results
├── p04.py                         # Model training script
│
├── Pt2/models/
│   ├── xgb_model.joblib           # Trained XGBoost model
│   └── xgb_scaler.joblib          # Feature scaler
│
├── Inference_Inputs/              # Daily race cards
├── Inference_Odds/                # Live odds data
├── Inference_Outputs/             # Predictions
├── Inference_Betting/             # Value bets
├── Inference_Logs/                # Processing logs
├── Inference_Actuals/             # Race results
└── Inference_Summary/             # Performance summaries
```

## 🧠 Feature Engineering

### Core Features (20 total)
1. **Form Indicators**
   - EMA_Form: Exponential moving average of recent performance
   - Recent_win_rate: Win rate over last 5 races
   - Career_win_rate: Lifetime win percentage

2. **Jockey-Trainer Stats**
   - Wilson_score: Confidence-adjusted win rate
   - jt_runs/jt_wins: Historical partnership performance

3. **Track Context**
   - Track_win_rate: Success rate at specific venue
   - Track_avg_distance: Average race distance at venue
   - Performance_on_going: Win rate on current ground condition

4. **Horse Attributes**
   - Age bins (young/prime/veteran)
   - Weight carried
   - Days since last race
   - Official rating

### Anti-Leakage Design
All features use **point-in-time calculations**:
- `groupby().cumcount()` for historical counts
- `groupby().cumsum() - current_value` for running totals
- Chronological sorting ensures no future data leaks

## 💰 Betting System

### Value Identification Criteria
- **Minimum Edge**: 1% (model probability - implied probability)
- **Minimum EV**: 1% expected value
- **Strong Bet**: 15%+ edge

### Kelly Criterion Staking
```python
Kelly = (bp - q) / b
Fractional Kelly = 0.25 * Kelly  # Quarter Kelly for safety
Stake = Fractional Kelly × Bankroll
```

### Trend Adjustments
- **Shortening odds**: +2% edge bonus (market confidence)
- **Drifting odds**: -2% edge penalty (market concerns)
- **Stable odds**: No adjustment

### Example Output
```
VALUE BETTING OPPORTUNITIES - 3 BETS
================================================================================
Horse                     Odds       Model P    Edge       EV         ROI%       Stake        Action      
--------------------------------------------------------------------------------
Thunder Bay              5/1        0.234      +0.067     +0.403     +40.3%     £6.25        STRONG BET
Silver Lining            7/2        0.198      +0.023     +0.095     +9.5%      £2.15        BET
Golden Arrow             4/1        0.214      +0.014     +0.071     +7.1%      £1.68        BET

SUMMARY:
  Total Bets: 3
  Strong Bets (Edge > 15%): 1
  Average Edge: 0.035 (3.5%)
  Total Kelly Stakes: £10.08
  Expected Profit: £2.27
```

## 📈 Performance Tracking

### Daily Summary
The audit script generates:
- **Prediction Performance**: Strike rate vs actual winners
- **Betting Performance**: Win rate, ROI, profit/loss
- **Detailed Bet Results**: Race-by-race breakdown

### Example Summary Log
```
PREDICTION PERFORMANCE
Valid Races: 42
Prediction Wins: 8
Strike Rate: 19.05%

BETTING PERFORMANCE
Races with Bets: 12
Bets Won: 3
Bet Win Rate: 25.00%
Total Staked: £47.50
Total Profit/Loss: £8.32
ROI: +17.52%
```

## 🔧 Model Training

### To Retrain Model
```bash
python p04.py
```

### Training Configuration
- **Algorithm**: XGBoost with histogram-based trees
- **Split**: 64% train / 16% validation / 20% test (chronological)
- **Hyperparameters**:
  - Learning rate: 0.01
  - Max depth: 5
  - Estimators: 780
  - Subsample: 0.8
  - Colsample: 0.8

### Data Requirements
- Historical race data: `merged_v2_horses_labeled.csv`
- Lookup table: `V2_engineered_features.csv`
- Minimum ~50,000 races for robust training

## ⚠️ Important Notes

### Web Scraping
- Racing Post structure may change - selectors in code may need updates
- Use polite delays (1-2s) between requests
- Selenium requires ChromeDriver installation

### Risk Management
- **Never bet more than you can afford to lose**
- Quarter Kelly is conservative; consider even smaller fractions
- Model confidence is **not** a guarantee of success
- Past performance ≠ future results

### Known Limitations
- Model struggles with horses having <2 career runs
- Abandoned races may cause data mismatches
- Odds scraping requires JavaScript execution (Selenium)

## 📝 Logging & Debugging

### Prediction Logs
Each inference run creates:
```
Inference_Logs/YYYY-MM-DD_logs.txt
```
Contains:
- Missing value summary
- Feature source tracking (lookup vs. calculated)
- Data quality warnings

### Common Issues

**"No input file found"**
- Ensure scraper ran successfully
- Check date format matches YYYY-MM-DD

**"NaN values detected"**
- Review logs for specific features
- Check if horse has sufficient history
- Verify date parsing in `process_features()`

**"No matching horses"**
- Horse name normalization may differ
- Check for typos in scraper output
- Verify case-insensitive matching

## 🤝 Contributing

When modifying scrapers:
1. Update `data-test-selector` attributes if site changes
2. Maintain chronological sorting for feature calculations
3. Add new features to `prepare_data_for_tabnet()`
4. Retrain model after significant feature changes

## 📜 License

This project is for **educational purposes only**. Use of this system for actual betting is at your own risk.

---

**Disclaimer**: This system is a research project. Horse racing betting involves substantial risk. The model's historical performance does not guarantee future results. Always gamble responsibly and within your means.