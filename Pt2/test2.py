#!/usr/bin/env python3
"""
bare_min_odds.py

Hardcoded version — just run `python bare_min_odds.py`

Computes:
 - Model-predicted positives (prob >= threshold)
 - True win rate among them (precision)
 - Bare-minimum decimal odds to break even = 1 / precision
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fractions import Fraction
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION (your real paths)
# =============================================================================
X_PATH = Path("Pt2/test_X.csv")
Y_PATH = Path("Pt2/test_Y.csv")
MODEL_PATH = Path("Pt2/models/xgb_model.joblib")
SCALER_PATH = Path("Pt2/models/xgb_scaler.joblib")

THRESHOLDS = [0.25, 0.5, 0.75]  # can change if you like
OUTPUT_PATH = Path("bare_min_odds_summary.csv")

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_data():
    X = pd.read_csv(X_PATH)
    y_df = pd.read_csv(Y_PATH, header=0)
    if isinstance(y_df, pd.DataFrame) and y_df.shape[1] == 1:
        y = y_df.iloc[:, 0]
    elif 'winner' in y_df.columns:
        y = y_df['winner']
    else:
        y = y_df.iloc[:, 0]
    return X, y.astype(int).values

def predict_proba(X):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X.values)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_scaled)[:, 1]
    elif hasattr(model, "predict"):
        preds = model.predict(X_scaled)
        if preds.ndim == 1:
            return preds
        return preds[:, 0]
    else:
        raise RuntimeError("Model has no predict_proba or predict method.")

def analyze(y_true, proba, thresholds):
    results = []
    for thresh in thresholds:
        mask = proba >= thresh
        predicted = mask.sum()
        if predicted == 0:
            precision = 0.0
            be_decimal = float('inf')
        else:
            tp = int(np.logical_and(mask, y_true == 1).sum())
            precision = tp / predicted
            be_decimal = 1.0 / precision if precision > 0 else float('inf')
        frac = str(Fraction(be_decimal).limit_denominator(100)) if np.isfinite(be_decimal) else "inf"
        results.append({
            "threshold": thresh,
            "predicted_count": int(predicted),
            "true_positives": int(np.logical_and(mask, y_true == 1).sum()),
            "precision_win_rate": precision,
            "bare_min_decimal_odds": be_decimal,
            "bare_min_fractional_like": frac
        })
    return pd.DataFrame(results)

# =============================================================================
# MAIN
# =============================================================================

def main():
    X, y = load_data()
    proba = predict_proba(X)
    summary = analyze(y, proba, THRESHOLDS)
    pd.set_option('display.float_format', lambda x: f"{x:.4f}" if np.isfinite(x) else "inf")
    print("\n=== Bare Minimum Odds Report ===")
    print(summary.to_string(index=False))
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved summary to {OUTPUT_PATH}")
    print("\nNote:")
    print(" precision_win_rate = true positives / model bets")
    print(" bare_min_decimal_odds = 1 / precision")
    print(" Example: precision = 0.40 → bare_min_decimal_odds = 2.50 → need odds > 2.5 for positive EV.")

if __name__ == "__main__":
    main()
