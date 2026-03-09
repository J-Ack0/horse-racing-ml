#!/usr/bin/env python3
"""
VOID___betting_filters.py -- hardened, corrected with updated output paths and L15 logic

Outputs:
- Inference_Bets/Bets_{date}.csv (picks with dashboard-compatible schema)
- Inference_Bets/Bets_L15/L15_{date}.csv (L15 recommendations)
- Classification_{date}.csv (detailed per-horse analysis)
"""

from __future__ import annotations
import argparse
from datetime import datetime, timedelta
import itertools
import math
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# ------------------------------- Utilities ---------------------------------

def fractional_to_decimal(frac_str: Any) -> float:
    if pd.isna(frac_str):
        return np.nan
    s = str(frac_str).strip()
    if s == '':
        return np.nan
    try:
        if '/' in s:
            parts = s.split('/')
            if len(parts) == 2:
                num = float(parts[0])
                den = float(parts[1])
                return (num / den) + 1.0
        return float(s)
    except Exception:
        return np.nan

def decimal_to_fractional(decimal_odds: float) -> str:
    """Convert decimal odds back to fractional format for display"""
    if pd.isna(decimal_odds) or decimal_odds <= 1.0:
        return "N/A"
    
    try:
        # Convert decimal to fractional (e.g., 3.5 -> 5/2)
        numerator = decimal_odds - 1.0
        
        # Common fractional odds
        fractions = {
            0.5: "1/2", 1.0: "Evens", 1.5: "6/4", 2.0: "2/1", 
            2.5: "5/2", 3.0: "3/1", 4.0: "4/1", 5.0: "5/1",
            6.0: "6/1", 7.0: "7/1", 8.0: "8/1", 9.0: "9/1",
            10.0: "10/1", 15.0: "15/1", 20.0: "20/1", 25.0: "25/1",
            33.0: "33/1", 50.0: "50/1", 100.0: "100/1"
        }
        
        if numerator in fractions:
            return fractions[numerator]
        
        # Try to find close match
        for frac_val, frac_str in fractions.items():
            if abs(numerator - frac_val) < 0.1:
                return frac_str
        
        # Otherwise return as X/1
        return f"{int(numerator)}/1"
    except Exception:
        return "N/A"

def safe_get_scalar(df: pd.DataFrame, idx: Any, col: str, default: Any = None):
    """
    Safely return a scalar value for df.loc[idx, col].
    Handles cases where duplicate column names cause a Series to be returned.
    """
    if col not in df.columns:
        return default
    try:
        val = df.loc[idx, col]
    except Exception:
        try:
            val = df.at[idx, col]
        except Exception:
            return default

    if isinstance(val, (pd.Series, pd.DataFrame)):
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:, 0]
        for element in val:
            if pd.notna(element):
                return element
        return default

    if pd.isna(val):
        return default
    return val

def find_optimal_threshold(threshold_results: pd.DataFrame, metric: str):
    if metric not in threshold_results.columns:
        raise KeyError(f"Metric {metric} not found")
    idx = threshold_results[metric].idxmax()
    row = threshold_results.loc[idx]
    return float(row['threshold']), row

# ------------------------- Single bet math ---------------------------------

def single_expected_profit_per_unit(p: float, d: float) -> float:
    return p * d - 1.0

def single_variance_per_unit(p: float, d: float) -> float:
    win_pay = (d - 1.0)
    lose_pay = -1.0
    mean = p * win_pay + (1 - p) * lose_pay
    e_sq = p * (win_pay ** 2) + (1 - p) * (lose_pay ** 2)
    var = e_sq - mean ** 2
    return max(var, 0.0)

def kelly_fraction(p: float, d: float) -> float:
    b = d - 1.0
    if b <= 0:
        return 0.0
    f = (p * d - 1.0) / b
    return max(0.0, f)

# ------------------------- Lucky15 math ------------------------------------

def lucky15_expected_return_and_variance(ps: List[float], ds: List[float], stake_per_line: float = 1.0):
    n = 4
    assert len(ps) == n and len(ds) == n, "Lucky15 requires 4 legs"
    outcome_results = []
    for mask in range(1 << n):
        prob = 1.0
        wins = []
        for i in range(n):
            if (mask >> i) & 1:
                prob *= ps[i]
                wins.append(i)
            else:
                prob *= (1 - ps[i])
        if prob == 0.0:
            continue
        k = len(wins)
        if k == 0:
            payout = 0.0
        else:
            payout = 0.0
            # singles
            for i in range(n):
                if i in wins:
                    payout += ds[i] * stake_per_line
            # doubles
            for a, b in itertools.combinations(range(n), 2):
                if a in wins and b in wins:
                    payout += (ds[a] * ds[b]) * stake_per_line
            # trebles
            for a, b, c in itertools.combinations(range(n), 3):
                if a in wins and b in wins and c in wins:
                    payout += (ds[a] * ds[b] * ds[c]) * stake_per_line
            # fourfold
            if len(wins) == 4:
                payout += (ds[0] * ds[1] * ds[2] * ds[3]) * stake_per_line
        profit = payout - (15 * stake_per_line)
        outcome_results.append({'prob': prob, 'payout': payout, 'profit': profit, 'k': k})
    exp_profit = sum(o['prob'] * o['profit'] for o in outcome_results)
    exp_gross_return = sum(o['prob'] * o['payout'] for o in outcome_results)
    exp_profit_sq = sum(o['prob'] * (o['profit'] ** 2) for o in outcome_results)
    var = max(0.0, exp_profit_sq - (exp_profit ** 2))
    std = math.sqrt(var)
    sharpe = (exp_profit / std) if std > 0 else 0.0
    return {
        'expected_net_profit': exp_profit,
        'expected_gross_return': exp_gross_return,
        'variance': var,
        'std_dev': std,
        'sharpe': sharpe,
        'per_line_stake': stake_per_line
    }

# -------------------- Betting classification / per-horse --------------------

def classify_single(p_est: float, d: float, bankroll: float, kelly_cap: float = 0.05, min_edge: float = 0.03):
    single_ev_unit = single_expected_profit_per_unit(p_est, d)
    var_unit = single_variance_per_unit(p_est, d)
    f_k = kelly_fraction(p_est, d)
    f_k_clamped = min(f_k, kelly_cap)
    suggested_stake_units = f_k_clamped * bankroll
    std_unit = math.sqrt(var_unit) if var_unit > 0 else 0.0
    sharpe_unit = (single_ev_unit / std_unit) if std_unit > 0 else 0.0
    
    # Calculate edge (implied prob vs model prob)
    implied_prob = 1.0 / d if d > 0 else 0.0
    edge = p_est - implied_prob
    
    if single_ev_unit > min_edge:
        classification = 'STRONG_BET_SINGLE'
        action = 'STRONG BET'
    elif single_ev_unit > 0:
        classification = 'BET_SINGLE'
        action = 'BET'
    else:
        classification = 'NO_BET'
        action = 'PASS'
    
    return {
        'p_est': p_est,
        'd': d,
        'edge': edge,
        'ev_per_unit': single_ev_unit,
        'var_per_unit': var_unit,
        'sharpe_unit': sharpe_unit,
        'kelly_full': f_k,
        'kelly_clamped': f_k_clamped,
        'suggested_stake_units': suggested_stake_units,
        'classification': classification,
        'action': action
    }

# --------------------- Threshold analysis save/print ------------------------

def print_threshold_analysis(threshold_results: pd.DataFrame, y_true, y_proba, output_dir: str = "Model_Thresholds"):
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(output_dir, f"xgb_model_t_{today}.csv")
    ordered_cols = ['threshold','precision','recall','f1_score','accuracy','specificity','tp','fp','tn','fn','predicted_positive']
    existing_cols = [c for c in ordered_cols if c in threshold_results.columns]
    save_df = threshold_results[existing_cols + [c for c in threshold_results.columns if c not in existing_cols]]
    save_df.to_csv(out_path, index=False)
    print("\nSaved threshold table to:", out_path)
    try:
        print("AUC-ROC:", roc_auc_score(y_true, y_proba))
    except Exception:
        pass
    return {}

# ------------------------- Main pipeline -----------------------------------

def build_and_evaluate(pred_path: str, odds_path: str, thr_path: str, bankroll: float = 100.0,
                       kelly_cap: float = 0.05, min_edge: float = 0.03,
                       top_candidates: int = 12, max_combos: int = 200, l15_sharpe_min: float = 0.1,
                       prob_threshold: float = 0.1):
    print("\nLoading files...")
    for p in (pred_path, odds_path, thr_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    df_pred = pd.read_csv(pred_path)
    df_odds = pd.read_csv(odds_path)
    df_thr = pd.read_csv(thr_path)
    
    # Convert fractional odds in df_odds
    for col in ['Odds_Live','odds_1','odds_2','odds_3','odds_4','odds_live']:
        if col in df_odds.columns:
            df_odds[f'decimal_{col.lower()}'] = df_odds[col].apply(fractional_to_decimal)
    
    # Normalize decimal_odds_live
    if 'decimal_odds_live' not in df_odds.columns:
        candidate_cols = [c for c in df_odds.columns if 'decimal' in c and 'live' in c]
        if candidate_cols:
            df_odds['decimal_odds_live'] = df_odds[candidate_cols[0]]
        elif 'Odds_Live' in df_odds.columns:
            df_odds['decimal_odds_live'] = df_odds['Odds_Live'].apply(fractional_to_decimal)
        elif 'odds_live' in df_odds.columns:
            df_odds['decimal_odds_live'] = df_odds['odds_live'].apply(fractional_to_decimal)
    
    # CLEANER MERGE LOGIC
    for d in [df_pred, df_odds]:
        if 'horse' in d.columns and 'horse_name' not in d.columns:
            d.rename(columns={'horse': 'horse_name'}, inplace=True)
        if 'track' in d.columns and 'track_name' not in d.columns:
            d.rename(columns={'track': 'track_name'}, inplace=True)
            
        if 'horse_name' in d.columns:
            d['horse_name'] = d['horse_name'].astype(str).str.strip().str.lower()
        if 'track_name' in d.columns:
            d['track_name'] = d['track_name'].astype(str).str.strip().str.lower()

    merge_keys = ['horse_name']
    if 'track_name' in df_pred.columns and 'track_name' in df_odds.columns:
        merge_keys.append('track_name')
    # Note: race_id will be created AFTER merge from track_name + time
        
    print(f"Merging on keys: {merge_keys}")
    df = pd.merge(df_pred, df_odds, on=merge_keys, how='inner', suffixes=('_pred', '_odds'))
    print(f"Merged {len(df)} rows correctly aligned.")
    
    # CREATE RACE_ID from track_name and time
    if 'track_name' in df.columns and 'time' in df.columns:
        df['race_id'] = df['track_name'].astype(str) + '_' + df['time'].astype(str)
        print(f"Created race_id from track_name and time")
    else:
        print("Warning: Could not create race_id - missing track_name or time columns")

    # Coerce numeric
    df['decimal_odds_live'] = pd.to_numeric(df['decimal_odds_live'], errors='coerce')
    if df['decimal_odds_live'].isna().any():
        def coerce_row_fractional(row):
            for c in ['Odds_Live','odds_live','odds_1','odds_2','odds_3','odds_4']:
                if c in row and pd.notna(row[c]):
                    v = fractional_to_decimal(row[c])
                    if not pd.isna(v):
                        return v
            return np.nan
        mask = df['decimal_odds_live'].isna()
        if mask.any():
            df.loc[mask, 'decimal_odds_live'] = df[mask].apply(coerce_row_fractional, axis=1)
    df['decimal_odds_live'] = pd.to_numeric(df['decimal_odds_live'], errors='coerce')
    
    bad_odds = df[df['decimal_odds_live'].isna()][['track_name','time','horse_name']].head(10) if set(['track_name','time','horse_name']).issubset(df.columns) else df[df['decimal_odds_live'].isna()].head(10)
    if len(bad_odds) > 0:
        print("Warning: some rows still have non-numeric odds (showing sample up to 10):")
        print(bad_odds.to_string(index=False))
    
    # Map probabilities to threshold rows
    def map_prob_to_threshold_estimate(p: float, thresholds_df: pd.DataFrame):
        if 'threshold' not in thresholds_df.columns:
            raise ValueError("thresholds_df must contain 'threshold'")
        thr = thresholds_df.sort_values('threshold').reset_index(drop=True)
        candidates = thr[thr['threshold'] <= p]
        row = candidates.iloc[-1] if len(candidates) > 0 else thr.iloc[0]
        return {
            'threshold': float(row.get('threshold', np.nan)),
            'precision': float(row.get('precision', np.nan)) if 'precision' in row.index else np.nan,
            'recall': float(row.get('recall', np.nan)) if 'recall' in row.index else np.nan,
            'f1': float(row.get('f1', np.nan)) if 'f1' in row.index else np.nan,
            'pred_plus': float(row.get('pred_plus', np.nan)) if 'pred_plus' in row.index else np.nan
        }
    
    mapped = df['win_probability'].apply(lambda p: map_prob_to_threshold_estimate(float(p), df_thr))
    mapped_df = pd.DataFrame(list(mapped))
    df = pd.concat([df.reset_index(drop=True), mapped_df.reset_index(drop=True)], axis=1)
    
    # Per-horse classification - HEAVILY WEIGHT MODEL PROBABILITY
    results = []
    for idx, row in df.iterrows():
        # Use win_probability primarily, only use precision as fallback
        model_prob = float(row['win_probability'])
        p_raw = row.get('precision', np.nan)
        
        # Weight model_prob heavily (90% model, 10% precision if available)
        if pd.notna(p_raw) and str(p_raw) != 'nan':
            precision_val = float(p_raw)
            p_est = 0.9 * model_prob + 0.1 * precision_val
        else:
            p_est = model_prob
        
        d_raw = row.get('decimal_odds_live', np.nan)
        if pd.isna(d_raw):
            d = 0.0
        else:
            try:
                d = float(d_raw)
            except Exception:
                parsed = fractional_to_decimal(d_raw)
                d = float(parsed) if not pd.isna(parsed) else 0.0
        
        single_metrics = classify_single(p_est, d, bankroll, kelly_cap=kelly_cap, min_edge=min_edge)
        merged = {**row.to_dict(), **single_metrics}
        results.append(merged)
    
    res_df = pd.DataFrame(results)
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_class = f'Classification_{date_str}.csv'
    res_df.to_csv(out_class, index=False)
    print(f"Saved per-horse classification to: {out_class}")
    
    # Create Bets CSV with DASHBOARD-COMPATIBLE SCHEMA
    bets_dir = 'Inference_Bets'
    os.makedirs(bets_dir, exist_ok=True)
    
    if 'classification' not in res_df.columns:
        res_df['classification'] = res_df.get('classification', 'NO_BET')
    
    picks = res_df[res_df['classification'].isin(['STRONG_BET_SINGLE','BET_SINGLE'])].copy()
    
    # Build dashboard-compatible CSV with required columns
    bets_output = []
    for idx, row in picks.iterrows():
        # Get original odds (fractional format for display)
        odds_live_frac = safe_get_scalar(picks, idx, 'Odds_Live', default=None) or \
                         safe_get_scalar(picks, idx, 'odds_live', default=None)
        
        if odds_live_frac is None or pd.isna(odds_live_frac):
            # Convert decimal back to fractional for display
            decimal_val = safe_get_scalar(picks, idx, 'decimal_odds_live', default=0.0)
            odds_live_frac = decimal_to_fractional(float(decimal_val))
        
        bets_output.append({
            'horse_name_odds': safe_get_scalar(picks, idx, 'horse_name', default='Unknown'),
            'track_name': safe_get_scalar(picks, idx, 'track_name', default='Unknown'),
            'time': safe_get_scalar(picks, idx, 'time', default=''),
            'odds_live': str(odds_live_frac),
            'win_probability': float(safe_get_scalar(picks, idx, 'win_probability', default=0.0)),
            'edge': float(safe_get_scalar(picks, idx, 'edge', default=0.0)),
            'ev': float(safe_get_scalar(picks, idx, 'ev_per_unit', default=0.0)),
            'kelly_stake': float(safe_get_scalar(picks, idx, 'suggested_stake_units', default=0.0)),
            'action': safe_get_scalar(picks, idx, 'action', default='BET'),
            'jockey': safe_get_scalar(picks, idx, 'jockey', default=''),
            'trainer': safe_get_scalar(picks, idx, 'trainer', default=''),
            'rating': safe_get_scalar(picks, idx, 'rating', default='')
        })
    
    bets_df = pd.DataFrame(bets_output)
    picks_out = os.path.join(bets_dir, f'Bets_{date_str}.csv')
    bets_df.to_csv(picks_out, index=False)
    print(f"Saved dashboard-compatible bets to: {picks_out} ({len(bets_df)} rows)")
    
    # Build L15 candidates - ONLY TOP HORSE UNLESS CLOSE TOGETHER
    if 'race_id' not in df.columns or df['race_id'].isna().all():
        print("Warning: No valid race_id found, cannot properly filter L15 candidates")
        eligible_for_l15 = df.copy()
    else:
        # Group by race_id and select top horses
        eligible_horses = []
        
        print(f"\nFiltering L15 candidates by race...")
        print(f"Total unique races: {df['race_id'].nunique()}")
        
        for race_id, race_group in df.groupby('race_id'):
            sorted_race = race_group.sort_values('win_probability', ascending=False)
            
            if len(sorted_race) >= 2:
                top_prob = sorted_race.iloc[0]['win_probability']
                second_prob = sorted_race.iloc[1]['win_probability']
                
                # If top 2 are within prob_threshold (0.1), include both
                if abs(top_prob - second_prob) <= prob_threshold:
                    eligible_horses.extend([sorted_race.iloc[0].name, sorted_race.iloc[1].name])
                    print(f"  {race_id}: Including top 2 horses (probs: {top_prob:.3f}, {second_prob:.3f})")
                else:
                    eligible_horses.append(sorted_race.iloc[0].name)
                    print(f"  {race_id}: Including top horse only (prob: {top_prob:.3f})")
            elif len(sorted_race) == 1:
                eligible_horses.append(sorted_race.iloc[0].name)
                print(f"  {race_id}: Single horse race")
        
        eligible_for_l15 = df.loc[eligible_horses]
        print(f"\nSelected {len(eligible_for_l15)} horses for L15 evaluation (top per race or close pairs)")
        print(f"From {df['race_id'].nunique()} unique races")
    
    top_pool = eligible_for_l15.sort_values('win_probability', ascending=False).head(top_candidates)
    
    out_l15 = []
    if len(top_pool) >= 4:
        combos = list(itertools.combinations(top_pool.index.tolist(), 4))
        combos = combos[:max_combos]
        for comb_idx, combo in enumerate(combos):
            ps = []
            ds = []
            names = []
            tracks = []
            history_counts = []
            bad_flag = False
            
            for i in combo:
                # Use win_probability primarily (weighted 90%)
                model_prob = safe_get_scalar(df, i, 'win_probability', default=0.0)
                prec = safe_get_scalar(df, i, 'precision', default=None)
                
                if prec is not None:
                    try:
                        prec_f = float(prec)
                        # 90% model, 10% precision
                        final_prob = 0.9 * float(model_prob) + 0.1 * prec_f
                    except Exception:
                        final_prob = float(model_prob)
                else:
                    final_prob = float(model_prob)
                
                ps.append(final_prob)
                
                # Odds
                d_val = safe_get_scalar(df, i, 'decimal_odds_live', default=None)
                if d_val is None or pd.isna(d_val):
                    d_val = safe_get_scalar(df, i, 'Odds_Live', default=None) or safe_get_scalar(df, i, 'odds_live', default=None)
                    d_parsed = fractional_to_decimal(d_val) if d_val is not None else np.nan
                    d_num = float(d_parsed) if not pd.isna(d_parsed) else 0.0
                else:
                    try:
                        d_num = float(d_val)
                    except Exception:
                        d_parsed = fractional_to_decimal(d_val)
                        d_num = float(d_parsed) if not pd.isna(d_parsed) else 0.0
                ds.append(d_num)
                
                # Names/tracks
                nm = safe_get_scalar(df, i, 'horse_name', default=None)
                if nm is None:
                    nm = safe_get_scalar(df, i, 'horse', default=str(i))
                tr = safe_get_scalar(df, i, 'track_name', default='')
                names.append(str(nm))
                tracks.append(str(tr))
                
                # History count
                hist = safe_get_scalar(df, i, 'history_count', default=0)
                try:
                    history_counts.append(int(hist))
                except Exception:
                    history_counts.append(0)
                
                if d_num == 0.0:
                    bad_flag = True
            
            if bad_flag:
                continue
            
            l15 = lucky15_expected_return_and_variance(ps, ds, stake_per_line=1.0)
            classification = 'NO_L15'
            if l15['expected_net_profit'] > 0 and l15['sharpe'] > l15_sharpe_min:
                classification = 'L15_STRONG'
            elif l15['expected_net_profit'] > 0:
                classification = 'L15_CANDIDATE'
            
            out_l15.append({
                'combo_index': comb_idx,
                'indices': combo,
                'names': '|'.join(names),
                'tracks': '|'.join(tracks),
                'history_count': '|'.join(map(str, history_counts)),
                'p_mean': float(np.mean(ps)),
                'd_mean': float(np.mean(ds)),
                'expected_net_profit': l15['expected_net_profit'],
                'expected_gross_return': l15['expected_gross_return'],
                'l15_std': l15['std_dev'],
                'l15_sharpe': l15['sharpe'],
                'classification': classification
            })
        
        # Create L15 output directory
        l15_dir = os.path.join('Inference_Bets', 'Bets_L15')
        os.makedirs(l15_dir, exist_ok=True)
        
        l15_df = pd.DataFrame(out_l15)
        l15_out_path = os.path.join(l15_dir, f'L15_{date_str}.csv')
        l15_df.to_csv(l15_out_path, index=False)
        print(f"Saved L15 candidate evaluations to: {l15_out_path} ({len(l15_df)} combos)")
    else:
        print("Not enough distinct candidates to evaluate L15 combos")
    
    print("\nPipeline complete.")

# ------------------------- CLI & defaults ----------------------------------

def main():
    TODAY =(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    default_pred = f"Inference_Outputs/Output_{TODAY}.csv"
    default_odds = f"Inference_Odds/Odds_{TODAY}.csv"
    default_thr = "Model_Thresholds/xgb_model_t_2026-01-14.csv"
    parser = argparse.ArgumentParser(description="Betting filters and L15 evaluator")
    parser.add_argument('--pred', default=default_pred)
    parser.add_argument('--odds', default=default_odds)
    parser.add_argument('--thresholds', default=default_thr)
    parser.add_argument('--bankroll', type=float, default=100.0)
    parser.add_argument('--kelly_cap', type=float, default=0.05)
    parser.add_argument('--min_edge', type=float, default=0.03)
    parser.add_argument('--top_candidates', type=int, default=12)
    parser.add_argument('--max_combos', type=int, default=200)
    parser.add_argument('--l15_sharpe_min', type=float, default=0.1)
    parser.add_argument('--prob_threshold', type=float, default=0.1, 
                       help='Max probability difference to include 2nd horse from race in L15')
    args = parser.parse_args()
    build_and_evaluate(
        pred_path=args.pred,
        odds_path=args.odds,
        thr_path=args.thresholds,
        bankroll=args.bankroll,
        kelly_cap=args.kelly_cap,
        min_edge=args.min_edge,
        top_candidates=args.top_candidates,
        max_combos=args.max_combos,
        l15_sharpe_min=args.l15_sharpe_min,
        prob_threshold=args.prob_threshold
    )

if __name__ == '__main__':
    main()