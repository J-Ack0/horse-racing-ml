# simulate_bets.py
import os
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Dict
import math

np.random.seed(42)

# ---------------------------
# Helpers / distributions
# ---------------------------
def generate_skewed_odds(n: int,
                         min_odds: float = 1.5,
                         max_odds: float = 40.0,
                         skew: float = 3.0) -> np.ndarray:
    """
    Generate 'n' decimal odds with a heavy concentration near min_odds (i.e., many near evens)
    skew > 1 pushes mass toward min_odds (i.e. many low odds, few longshots).
    Returns decimal odds (e.g. 2.0 = evens).
    """
    # uniform^skew -> skewing toward 0 for skew>1
    u = np.random.rand(n) ** skew
    odds = min_odds + (max_odds - min_odds) * u
    odds = np.clip(odds, min_odds, max_odds)
    return odds

def implied_prob_from_decimal(odds: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """
    Convert decimal odds to implied probability. margin can be >0 to simulate bookie margin.
    implied = 1 / odds, optionally reduce them to include margin across market.
    Simple approach here: implied = 1/odds, then scale up by margin factor if requested.
    """
    ip = 1.0 / odds
    if margin > 0:
        # naive margin: inflate implied probabilities proportionally so they sum > 1
        # but since single selection per race here, just multiply by (1 + margin)
        ip = ip * (1 + margin)
        ip = np.clip(ip, 1e-6, 1 - 1e-6)
    return ip

# Kelly fraction (full Kelly) for decimal odds:
# fraction = (p*(b+1) - 1) / b  where b = odds - 1
def kelly_fraction(p: np.ndarray, decimal_odds: np.ndarray) -> np.ndarray:
    b = decimal_odds - 1.0
    # protect division by very small numbers
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = (p * (b + 1.0) - 1.0) / b
    frac = np.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0)
    # set negative fractions to 0 (no bet)
    frac[frac < 0] = 0.0
    return frac

# ---------------------------
# Loading / prediction
# ---------------------------
def load_test_data(x_path='test_X.csv', y_path='test_Y.csv') -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path, header=0)
    # y could be a single-column DF or Series
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    return X, y.astype(int)

def get_model_predictions(X: pd.DataFrame,
                          model_path: str = "Pt2/models/xgb_model.joblib",
                          scaler_path: str = "Pt2/models/xgb_scaler.joblib"):
    """Try to load model+scaler to produce real predictions. Returns None if files not present."""
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(X.values)
            proba = model.predict_proba(X_scaled)[:, 1]
            print("Loaded model and scaler - using real model probabilities.")
            return proba
        except Exception as e:
            print("Model load failed:", e)
    return None

def simulate_model_proba(y_true: np.ndarray, signal_strength: float = 0.05) -> np.ndarray:
    """
    Create a simulated model prediction probability vector.
    signal_strength in [0,1] controls how well the simulated model correlates with truth.
    - 0 => random noise (no skill).
    - 1 => very strong signal (almost perfect).
    NOTE: This intentionally uses true labels to simulate 'skill' for backtesting.
    """
    n = len(y_true)
    base = np.random.beta(1.2, 8.0, n)  # mostly small probabilities
    # bump winners' probabilities by a scaled amount
    proba = base * (1 - signal_strength) + signal_strength * (0.2 + 0.75 * y_true)
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    return proba

# ---------------------------
# Betting strategies
# ---------------------------
# ---------------------------
# Patch: safer simulate_strategy + bootstrap runner + plotting
# ---------------------------
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

# Ensure output dir exists
os.makedirs("Pt2", exist_ok=True)

def simulate_strategy_safe(y_true: np.ndarray,
                           model_proba: np.ndarray,
                           odds: np.ndarray,
                           strategy: str,
                           init_bankroll: float = 100.0,
                           flat_stake: float = 10.0,
                           threshold: float = 0.4,
                           kelly_frac: float = 0.1,             # fraction of full Kelly to use
                           per_bet_bankroll_cap: float = 0.02, # max fraction of bankroll per bet
                           absolute_stake_cap: float = None,
                           max_bankroll_for_sim: float = 1e9,  # safety cap
                           tiny_stake_floor: float = 1e-6
                           ) -> Tuple[pd.DataFrame, Dict]:
    """
    Safer simulate_strategy:
      - fractional Kelly (kelly_frac)
      - per-bet bankroll cap (per_bet_bankroll_cap)
      - optional absolute stake cap
      - numeric safety stop if bankroll exceeds max_bankroll_for_sim
    """
    n = len(y_true)
    bankroll = float(init_bankroll)
    peak = bankroll
    max_dd = 0.0
    rows = []
    implied = implied_prob_from_decimal(odds)

    for i in range(n):
        if not np.isfinite(bankroll) or bankroll <= 0:
            # bankrupt or numeric failure -> stop placing bets
            rows.append({
                'index': i, 'model_p': float(model_proba[i]), 'odds': float(odds[i]),
                'implied_p': float(implied[i]), 'true': int(y_true[i]),
                'stake': 0.0, 'profit': 0.0, 'bankroll': bankroll, 'max_drawdown': max_dd
            })
            continue

        p_hat = float(model_proba[i])
        dec_odds = float(odds[i])
        imp = float(implied[i])
        true_win = int(y_true[i])
        stake = 0.0

        if strategy == 'flat_every':
            stake = flat_stake
        elif strategy == 'threshold_flat':
            if p_hat >= threshold:
                stake = flat_stake
        elif strategy in ('kelly_full', 'kelly_fractional'):
            b = dec_odds - 1.0
            if b <= 0:
                kf_full = 0.0
            else:
                kf_full = (p_hat * (b + 1.0) - 1.0) / b
                kf_full = max(0.0, kf_full)
            # Apply user fractional Kelly multiplier
            effective_kf = kf_full * float(kelly_frac)
            stake = bankroll * effective_kf
        elif strategy == 'edge_prop':
            edge = p_hat - imp
            if edge > 0:
                stake = edge * per_bet_bankroll_cap * bankroll / (1 - imp + 1e-9)
            else:
                stake = 0.0
        else:
            raise ValueError("Unknown strategy")

        # Apply caps
        stake = max(0.0, stake)
        stake_cap_from_bankroll = per_bet_bankroll_cap * bankroll
        stake = min(stake, stake_cap_from_bankroll)
        if absolute_stake_cap is not None:
            stake = min(stake, absolute_stake_cap)
        if stake < tiny_stake_floor:
            stake = 0.0

        profit = 0.0
        if stake > 0:
            if true_win == 1:
                profit = stake * (dec_odds - 1.0)
            else:
                profit = -stake
            bankroll += profit

            # Safety numeric clamp
            if not np.isfinite(bankroll) or bankroll > max_bankroll_for_sim:
                bankroll = min(bankroll if np.isfinite(bankroll) else 0.0, max_bankroll_for_sim)
                rows.append({
                    'index': i, 'model_p': p_hat, 'odds': dec_odds, 'implied_p': imp,
                    'true': true_win, 'stake': stake, 'profit': profit,
                    'bankroll': bankroll, 'max_drawdown': max_dd
                })
                # Print a one-line safety message and stop simulation
                print(f"[safety stop] bet={i} bankroll clipped to {bankroll}")
                break

            peak = max(peak, bankroll)
            dd = peak - bankroll
            max_dd = max(max_dd, dd)

        rows.append({
            'index': i,
            'model_p': p_hat,
            'odds': dec_odds,
            'implied_p': imp,
            'true': true_win,
            'stake': stake,
            'profit': profit,
            'bankroll': bankroll,
            'max_drawdown': max_dd
        })

    df = pd.DataFrame(rows)
    total_staked = df['stake'].sum()
    total_profit = df['profit'].sum()
    roi = total_profit / total_staked if total_staked > 0 else float('nan')
    bets_placed = int((df['stake'] > 0).sum())
    wins = int(((df['stake'] > 0) & (df['true'] == 1)).sum())
    summary = {
        'strategy': strategy,
        'init_bankroll': init_bankroll,
        'final_bankroll': float(bankroll),
        'total_profit': float(total_profit),
        'total_staked': float(total_staked),
        'roi': float(roi),
        'bets_placed': bets_placed,
        'wins': wins,
        'win_rate_on_bets': (wins / bets_placed) if bets_placed > 0 else float('nan'),
        'max_drawdown': float(max_dd)
    }
    return df, summary

def run_bootstrap_simulations(x_path='test_X.csv',
                              y_path='test_Y.csv',
                              try_load_model: bool = True,
                              signal_strength: float = 0.05,
                              odds_skew: float = 3.0,
                              min_odds: float = 1.5,
                              max_odds: float = 40.0,
                              strategy: str = 'kelly_fractional',
                              n_runs: int = 100,
                              kelly_frac: float = 0.1,
                              per_bet_bankroll_cap: float = 0.02,
                              absolute_stake_cap: float = None,
                              max_bankroll_for_sim: float = 1e9,
                              jitter_model_sigma: float = 0.01,
                              seed: int = 42):
    """
    Run multiple randomized simulations to estimate expected yield and risk.
    - If a real model is loaded, uses same model_proba but adds Gaussian jitter per run
      to represent uncertainty (keeps correlation to truth).
    - If using simulated model_proba (no real model found), resamples a new simulated proba each run.
    Returns summary DataFrame (one row per run) and also saves plots.
    """
    X, y = load_test_data(x_path, y_path)
    n = len(y)
    rng = np.random.RandomState(seed)
    base_model_proba = None
    if try_load_model:
        base_model_proba = get_model_predictions(X)
    results = []
    all_trajectories = []

    for r in tqdm(range(n_runs), desc="Bootstrap runs"):
        # generate odds deterministic per run (or randomized if you prefer)
        odds = generate_skewed_odds(n, min_odds=min_odds, max_odds=max_odds, skew=odds_skew)

        # Decide model probabilities for this run
        if base_model_proba is not None:
            # jitter the real model probabilities to reflect parameter uncertainty
            jitter = rng.normal(0.0, jitter_model_sigma, size=len(base_model_proba))
            model_proba = np.clip(base_model_proba + jitter, 1e-6, 1 - 1e-6)
        else:
            # resimulate a noisy model proba each run
            model_proba = simulate_model_proba(y.values, signal_strength=signal_strength)
            model_proba = np.clip(model_proba + rng.normal(0.0, jitter_model_sigma, size=n), 1e-6, 1 - 1e-6)

        df_traj, summary = simulate_strategy_safe(
            y_true=y.values,
            model_proba=model_proba,
            odds=odds,
            strategy=strategy,
            init_bankroll=1000.0,
            flat_stake=10.0,
            threshold=0.25,
            kelly_frac=kelly_frac,
            per_bet_bankroll_cap=per_bet_bankroll_cap,
            absolute_stake_cap=absolute_stake_cap,
            max_bankroll_for_sim=max_bankroll_for_sim
        )
        results.append(summary)
        # pad/truncate bankroll series to same length for aggregation
        all_trajectories.append(df_traj['bankroll'].values)

    # Normalize trajectories to equal length by padding final bankroll forward
    max_len = max(len(t) for t in all_trajectories)
    padded = np.zeros((len(all_trajectories), max_len))
    for i, t in enumerate(all_trajectories):
        padded[i, :len(t)] = t
        if len(t) < max_len:
            padded[i, len(t):] = t[-1]  # carry-forward final bankroll

    # Compute percentiles over time
    pctl = {q: np.percentile(padded, q, axis=0) for q in [5, 25, 50, 75, 95]}
    # Save trajectory plot
    plt.figure(figsize=(12, 6))
    xs = np.arange(padded.shape[1])
    plt.fill_between(xs, pctl[5], pctl[95], alpha=0.15, label='5-95 pct')
    plt.fill_between(xs, pctl[25], pctl[75], alpha=0.25, label='25-75 pct')
    plt.plot(xs, pctl[50], linewidth=2, label='median')
    plt.xlabel('Bet index (chronological)')
    plt.ylabel('Bankroll')
    plt.title(f'Bootstrap bankroll trajectories ({n_runs} runs) - strategy={strategy}')
    plt.legend()
    traj_path = os.path.join("Pt2", "sim_bankroll_trajectories.png")
    plt.savefig(traj_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Final bankroll histogram
    final_bankrolls = np.array([s['final_bankroll'] for s in results])
    plt.figure(figsize=(10, 5))
    plt.hist(final_bankrolls, bins=50)
    plt.xlabel('Final Bankroll')
    plt.ylabel('Count')
    plt.title(f'Final bankroll distribution ({n_runs} runs) - strategy={strategy}')
    hist_path = os.path.join("Pt2", "sim_final_bankroll_hist.png")
    plt.savefig(hist_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Summarize runs
    res_df = pd.DataFrame(results)
    summary_stats = {
        'runs': n_runs,
        'mean_final_bankroll': res_df['final_bankroll'].mean(),
        'median_final_bankroll': res_df['final_bankroll'].median(),
        'p5_final_bankroll': res_df['final_bankroll'].quantile(0.05),
        'p25_final_bankroll': res_df['final_bankroll'].quantile(0.25),
        'p75_final_bankroll': res_df['final_bankroll'].quantile(0.75),
        'p95_final_bankroll': res_df['final_bankroll'].quantile(0.95),
        'mean_roi': res_df['roi'].replace([np.inf, -np.inf], np.nan).mean(),
        'mean_bets_placed': res_df['bets_placed'].mean()
    }

    # Print summary
    print("\nBOOTSTRAP SUMMARY:")
    for k, v in summary_stats.items():
        if isinstance(v, float):
            print(f"  {k:25}: {v:,.4f}")
        else:
            print(f"  {k:25}: {v}")

    print(f"\nSaved trajectory plot -> {traj_path}")
    print(f"Saved final bankroll histogram -> {hist_path}")

    return res_df, padded, pctl

# ---------------------------
# Example patch-run: call from __main__ or interactive session
# ---------------------------
if __name__ == "__main__":
    # Parameters you can tweak
    strategy_to_test = 'kelly_fractional'  # 'threshold_flat', 'flat_every', 'edge_prop', etc.
    runs = 100
    res_df, padded, pctl = run_bootstrap_simulations(
        x_path='test_X.csv',
        y_path='test_Y.csv',
        try_load_model=True,        # use your real model if present
        signal_strength=0.08,      # used only if no real model
        odds_skew=3.5,
        min_odds=1.5,
        max_odds=50.0,
        strategy=strategy_to_test,
        n_runs=runs,
        kelly_frac=0.10,           # start with 10% Kelly
        per_bet_bankroll_cap=0.02, # 2% bankroll cap per bet
        absolute_stake_cap=2000.0, # optional absolute cap
        max_bankroll_for_sim=1e9,  # safety stop
        jitter_model_sigma=0.01,
        seed=42
    )

    # Show small summary table
    print("\nSAMPLE RUNS (first 6):")
    print(res_df.head(6).to_string(index=False))
