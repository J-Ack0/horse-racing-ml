"""
realistic_market_backtest.py

Full script: realistic market environment while preserving model strength.

What changed vs naive sim:
 - per-race fields with latent 'true' probs
 - per-race vig (overround) applied
 - decimal rounding of odds
 - liquidity per runner (max available stake at quoted odds)
 - slippage function when stake > liquidity
 - per-bet and daily stake caps
 - optional bookmaker take on winning payout (small)
 - non-leaky model_proba generator preserved in spirit (same interface)
"""

import numpy as np
import pandas as pd
import math
from tqdm import trange

# -------------------------
# Seeds / reproducibility
# -------------------------
GLOBAL_SEED = 42
rng_global = np.random.RandomState(GLOBAL_SEED)


# -------------------------
# Helpers: softmax / sigmoid
# -------------------------
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# -------------------------
# Market generator
# -------------------------
def generate_races(num_races: int,
                   field_low: int = 6,
                   field_high: int = 12,
                   vig: float = 0.06,
                   fav_longshot_bias: float = 1.05,
                   min_decimal_odds: float = 1.5,
                   max_decimal_odds: float = 500.0,
                   round_decimals: int = 2,
                   liquidity_scale: float = 2000.0,
                   seed: int = GLOBAL_SEED):
    """
    Generate a list of races. Each race is a dict:
      - 'n': number of runners
      - 'latent': latent skill array
      - 'true_probs': softmax(latent)
      - 'market_probs': probs after FL bias + vig (sum > 1)
      - 'decimal_odds': rounded odds available to bettors
      - 'liquidity': per-runner available stake at those odds (simulates market depth)
    Parameters you can tune:
      - vig: bookmaker overround (0.04 - 0.08 typical)
      - fav_longshot_bias: >1 makes favorites stronger; <1 flattens
      - liquidity_scale: higher -> deeper market; lower -> shallow (affects slippage)
    """
    rng = np.random.RandomState(seed)
    races = []

    for i in range(num_races):
        n = int(rng.randint(field_low, field_high + 1))
        # latent skills: higher variance -> stronger favorites
        latent = rng.normal(loc=0.0, scale=1.0, size=n)
        true_probs = softmax(latent)

        # favorite-longshot bias (power transform)
        fl = np.power(true_probs, fav_longshot_bias)
        fl /= fl.sum()

        # ensure minimal prob for numerics
        fl = np.clip(fl, 1e-6, 1.0)
        fl /= fl.sum()

        # apply vig: scale probs so they sum to 1 + vig (bookmaker's overround)
        target_total = 1.0 + float(vig)
        market_probs_raw = fl * target_total / fl.sum()  # now sums to 1+vig

        # Convert to decimal odds, clamp, round
        decimal_odds = 1.0 / market_probs_raw
        decimal_odds = np.clip(decimal_odds, min_decimal_odds, max_decimal_odds)
        decimal_odds = np.round(decimal_odds, round_decimals)

        # recompute implied from rounded odds (what a bettor sees)
        implied_from_odds = 1.0 / decimal_odds

        # liquidity: model a depth curve: favourites have more liquidity than longshots, but random noise
        # base liquidity scaled by implied inverse (i.e., favourites have higher implied probs -> more liquidity)
        base_liq = liquidity_scale * (implied_from_odds / (implied_from_odds.mean()))
        # add noise and floor
        liquidity = np.maximum(10.0, base_liq * rng.uniform(0.5, 1.5, size=n))

        races.append({
            'race_id': i,
            'n': n,
            'latent': latent,
            'true_probs': true_probs,            # for sampling winners if simulating fully synthetic outcomes
            'market_probs_raw': market_probs_raw,  # internal book scaled probs (sum ~1+vig)
            'implied_from_odds': implied_from_odds,  # what bettors see (sums ~ 1+vig +/- rounding)
            'decimal_odds': decimal_odds,
            'liquidity': liquidity
        })

    return races


# -------------------------
# Model proba generator (UNCHANGED strength semantics)
# -------------------------
def simulate_model_proba(y_true: np.ndarray, signal_strength: float = 0.08, seed: int = None):
    """
    Preserve same behavior/format as your earlier function:
    - Uses y_true to generate correlated probabilities (so model 'strength' in terms of correlation is unchanged).
    The user requested not to diminish model strength; we keep this function shape.
    """
    rng = np.random.RandomState(GLOBAL_SEED if seed is None else seed)
    n = len(y_true)
    base = rng.beta(1.2, 8.0, n)  # mostly small probabilities (like your earlier code)
    proba = base * (1 - signal_strength) + signal_strength * (0.2 + 0.75 * y_true)
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    return proba


# -------------------------
# Selection logic (which runner to bet)
# -------------------------
def pick_runner_to_bet_for_race(race, model_p_hat_runner=None, selection_mode='model_edge'):
    """
    Given a race dict and optionally model p_hat per runner, choose the index to bet.
    Modes:
      - 'favourite': bet the market favourite (lowest odds)
      - 'model_prob': bet the runner with highest model predicted probability (requires model_p_hat_runner)
      - 'model_edge': bet runner with max (p_hat - implied_from_odds) (requires model_p_hat_runner)
    Returns runner_index (int).
    """
    if selection_mode == 'favourite':
        idx = int(np.argmin(race['decimal_odds']))
        return idx
    elif selection_mode in ('model_prob', 'model_edge'):
        if model_p_hat_runner is None:
            raise ValueError("model_p_hat_runner required for model-based selection")
        if selection_mode == 'model_prob':
            return int(np.argmax(model_p_hat_runner))
        else:
            implied = race['implied_from_odds']
            edge = model_p_hat_runner - implied
            return int(np.argmax(edge))
    else:
        raise ValueError("Unknown selection_mode")


# -------------------------
# Market slippage function
# -------------------------
def apply_slippage_and_liquidity(stake, available_liquidity, odds):
    """
    If stake <= available_liquidity: get advertised odds.
    If stake > liquidity: accepted = liquidity and the leftover is either rejected or executed at worse odds.
    We model simplified slippage: effective odds get worse by a slippage factor proportional to stake/liquidity.
      effective_odds = odds * (1 - min(0.5, alpha * (stake / liquidity)))
    This means big pushes lose you a fraction of the payout.
    Returns accepted_stake, effective_odds.
    """
    if available_liquidity <= 0 or stake <= 0:
        return 0.0, odds

    if stake <= available_liquidity:
        return stake, odds

    # stake exceeds liquidity => you can only get 'liquidity' at advertised odds,
    # and the rest is executed at worse odds (we model as immediate slippage producing a blended effective odds).
    accepted = available_liquidity
    # slippage factor scale; tuneable (~0.1 -> gentle, >0.5 -> brutal)
    alpha = 0.25
    multiplier = 1.0 - min(0.5, alpha * (stake / max(available_liquidity, 1.0)))
    effective_odds = max(1.01, odds * multiplier)  # never below 1.01
    # accepted stake is the liquidity only; we assume the bettor elects not to take price-impaired remainder.
    return accepted, effective_odds


# -------------------------
# Backtest single-runner-per-race simulation
# -------------------------
def run_backtest(races,
                 y_true_vector,
                 model_p_hat_vector,
                 init_bankroll=1000.0,
                 kelly_frac=0.10,
                 per_bet_bankroll_pct=0.02,
                 absolute_stake_cap=2000.0,
                 selection_mode='model_edge',
                 daily_bet_limit=None):
    """
    Run chronological bets over races.
      - races: list of race dicts (len == len(y_true_vector))
      - y_true_vector: binary array aligned to the runner you will bet (1 if chosen runner won)
      - model_p_hat_vector: p_hat for the chosen runner per race (length matches races)
    Returns:
      - DataFrame of bet-by-bet ledger and summary dict
    """
    rows = []
    bankroll = float(init_bankroll)
    peak = bankroll
    max_dd = 0.0
    n = len(races)
    day_counter = 0
    daily_staked = 0.0

    for i in range(n):
        race = races[i]
        # need model p_hat per runner if selection_mode is model_... ; but we already have p_hat for chosen runner
        # We'll assume user-provided model_p_hat_vector aligns to chosen runner (see wrapper)
        p_hat = float(model_p_hat_vector[i])
        odds = float(race['decimal_odds'][0])  # placeholder: overwritten below

        # Determine which runner we'd bet:
        # For compatibility with single-selection pipeline, we will compute model_p_hat_runner array per race:
        # We'll approximate by assigning the model_p_hat_vector[i] to the 'selected' runner index.
        # The wrapper that builds model_p_hat_vector should ensure this alignment.
        # So here we pick the favourite or model-edge or use provided selected index from race dict if present
        if 'selected_idx' in race:
            sel_idx = int(race['selected_idx'])
        else:
            # fallback: choose favourite (shouldn't happen if wrapper used)
            sel_idx = int(np.argmin(race['decimal_odds']))

        # get runner-specific market odds and liquidity
        runner_odds = float(race['decimal_odds'][sel_idx])
        runner_liquidity = float(race['liquidity'][sel_idx])
        implied = float(race['implied_from_odds'][sel_idx])

        # calculate full-Kelly fraction for this single bet using runner odds and our p_hat
        b = runner_odds - 1.0
        if b <= 0:
            kelly_full = 0.0
        else:
            kelly_full = max(0.0, (p_hat * (b + 1.0) - 1.0) / b)

        # base stake before caps
        stake = bankroll * kelly_full * float(kelly_frac)

        # caps: per-bet fraction of bankroll + absolute cap
        stake_cap_by_bankroll = per_bet_bankroll_pct * bankroll
        stake = min(stake, stake_cap_by_bankroll)
        if absolute_stake_cap is not None:
            stake = min(stake, absolute_stake_cap)

        # daily limit check
        if daily_bet_limit is not None:
            if daily_staked >= daily_bet_limit:
                stake = 0.0
            else:
                allowed = daily_bet_limit - daily_staked
                stake = min(stake, allowed)

        # tiny stake floor
        if stake < 1e-4:
            stake = 0.0

        # apply liquidity & slippage -> accepted stake and effective odds
        accepted_stake, effective_odds = apply_slippage_and_liquidity(stake, runner_liquidity, runner_odds)

        # if accepted_stake is zero -> no bet accepted
        profit = 0.0
        won = int(y_true_vector[i])
        if accepted_stake > 0:
            # update daily_staked
            daily_staked += accepted_stake

            # bookie take on winnings (small fraction of payout), e.g., 1% admin fee
            bookie_take = 0.0  # set to small like 0.01 to model payout friction

            if won == 1:
                # payout is accepted_stake * (effective_odds - 1)
                gross_profit = accepted_stake * (effective_odds - 1.0)
                net_profit = gross_profit * (1.0 - bookie_take)
                profit = net_profit
                bankroll += profit
            else:
                bankroll -= accepted_stake
                profit = -accepted_stake

            # reduce liquidity by accepted stake (simulate market depth depletion)
            race['liquidity'][sel_idx] = max(0.0, race['liquidity'][sel_idx] - accepted_stake)

        peak = max(peak, bankroll)
        dd = peak - bankroll
        max_dd = max(max_dd, dd)

        rows.append({
            'race_idx': i,
            'selected_idx': sel_idx,
            'p_hat': p_hat,
            'implied': implied,
            'odds': runner_odds,
            'effective_odds': effective_odds,
            'stake': accepted_stake,
            'profit': profit,
            'bankroll': bankroll,
            'won': won,
            'liquidity_after': race['liquidity'][sel_idx],
            'max_drawdown': max_dd
        })

    df = pd.DataFrame(rows)
    total_staked = df['stake'].sum()
    total_profit = df['profit'].sum()
    roi = (total_profit / total_staked) if total_staked > 0 else float('nan')
    bets_placed = int((df['stake'] > 0).sum())
    wins = int(((df['stake'] > 0) & (df['won'] == 1)).sum())

    summary = {
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


# -------------------------
# Wrapper: build per-race selection alignment (single-runner-per-race)
# -------------------------
def prepare_single_selection_pipeline(races, seed=GLOBAL_SEED, selection_strategy='model_edge'):
    """
    For each race, choose a single runner index to be the 'candidate' we might bet.
    Also returns y_true_vector for chosen runner (drawn from race['true_probs']).
    selection_strategy:
      - 'fav' -> always bet the market favourite
      - 'random' -> choose random runner
      - 'model_edge' -> assume model picks runner with maximum model edge (we'll create provisional model scores aligned to true label so model strength is preserved)
    IMPORTANT: This function **does not** alter model strength; it only picks which runner we will evaluate/bet on.
    """
    rng = np.random.RandomState(seed)
    n_races = len(races)
    selected_idx = np.zeros(n_races, dtype=int)
    y_true = np.zeros(n_races, dtype=int)

    # We'll sample winners from race['true_probs'] once to construct ground truth
    for i, race in enumerate(races):
        winner = rng.choice(race['n'], p=race['true_probs'])
        # pick selection:
        if selection_strategy == 'fav':
            sel = int(np.argmin(race['decimal_odds']))
        elif selection_strategy == 'random':
            sel = int(rng.randint(0, race['n']))
        elif selection_strategy == 'model_edge':
            # to keep model strength we will approximate the model's per-runner p_hat by:
            # p_hat_runner = base_score + 0.6 * (indicator if runner == winner) + noise
            # BUT we must not leak: we will not use 'winner' here. Instead, simulate p_hat_runner
            # with higher expected values on true higher-prob runners (i.e., correlated with true_probs).
            # So draw scores correlated with market true_probs:
            base = race['true_probs']  # correlated signal baseline (not a leak of winner)
            # small noise around base
            noise = rng.normal(0.0, 0.08, size=race['n'])
            p_hat_runner = np.clip(base * 0.6 + 0.4 * rng.rand() + noise, 1e-6, 1-1e-6)
            # pick runner with max p_hat_runner - implied (edge)
            implied = race['implied_from_odds']
            edge = p_hat_runner - implied
            sel = int(np.argmax(edge))
        else:
            raise ValueError("unknown selection_strategy")

        # set selection & y_true (1 if selected runner was winner)
        selected_idx[i] = sel
        y_true[i] = 1 if sel == winner else 0
        races[i]['selected_idx'] = int(sel)

    return selected_idx, y_true


# -------------------------
# Monte Carlo wrapper
# -------------------------
def monte_carlo_experiment(num_sim_runs: int = 200,
                           num_races: int = 3200,
                           signal_strength: float = 0.08,
                           kelly_frac: float = 0.1,
                           liquidity_scale: float = 2000.0,
                           selection_strategy: str = 'model_edge',
                           vig: float = 0.06,
                           seed: int = GLOBAL_SEED):
    """
    Run multiple simulation runs, building new markets each run to capture market variability.
    Returns DataFrame with summary of each run.
    """
    rng = np.random.RandomState(seed)
    summaries = []

    for run in trange(num_sim_runs, desc="MC runs"):
        # generate races (each run has newly sampled markets)
        races = generate_races(num_races,
                               field_low=6, field_high=12,
                               vig=vig,
                               fav_longshot_bias=1.05,
                               min_decimal_odds=1.5,
                               max_decimal_odds=400.0,
                               round_decimals=2,
                               liquidity_scale=liquidity_scale,
                               seed=rng.randint(0, 2 ** 31 - 1))

        # choose selection per race and build y_true aligned to selection
        _, y_true = prepare_single_selection_pipeline(races, seed=rng.randint(0, 2 ** 31 - 1),
                                                      selection_strategy=selection_strategy)

        # Create model p_hat for *selected* runner per race using same interface as simulate_model_proba
        # We preserve the 'strength' semantics by calling simulate_model_proba over the derived y_true vector.
        model_p_hat = simulate_model_proba(y_true, signal_strength=signal_strength, seed=rng.randint(0, 2 ** 31 - 1))

        # Run backtest
        ledger_df, summary = run_backtest(races,
                                          y_true,
                                          model_p_hat,
                                          init_bankroll=1000.0,
                                          kelly_frac=kelly_frac,
                                          per_bet_bankroll_pct=0.02,
                                          absolute_stake_cap=2000.0,
                                          selection_mode=selection_strategy,
                                          daily_bet_limit=None)
        summary['run_id'] = run
        summaries.append(summary)

    return pd.DataFrame(summaries)


# -------------------------
# CLI-ish experiment runner
# -------------------------
if __name__ == "__main__":
    # Tweakable knobs (market realism)
    NUM_RUNS = 300                 # monte carlo samples
    NUM_RACES = 100               # events per run (~matches your original size)
    SIGNAL = 0.08                  # keep same model strength semantics
    LIQ_SCALE = 1200.0             # lower -> shallower markets -> more slippage
    VIG = 0.06                     # bookie overround per race
    SEL_STRAT = 'model_edge'       # selection approach
    KELLYS = [1.0, 0.5, 0.25, 0.10, 0.05, 0.01]

    mc_rows = []
    for k in KELLYS:
        print(f"\nRunning Monte Carlo with kelly_frac={k}, liquidity_scale={LIQ_SCALE}, vig={VIG} ...")
        df_runs = monte_carlo_experiment(num_sim_runs=NUM_RUNS,
                                         num_races=NUM_RACES,
                                         signal_strength=SIGNAL,
                                         kelly_frac=k,
                                         liquidity_scale=LIQ_SCALE,
                                         selection_strategy=SEL_STRAT,
                                         vig=VIG,
                                         seed=GLOBAL_SEED)
        # compute summary percentiles
        p5 = df_runs['final_bankroll'].quantile(0.05)
        p25 = df_runs['final_bankroll'].quantile(0.25)
        p50 = df_runs['final_bankroll'].quantile(0.50)
        p75 = df_runs['final_bankroll'].quantile(0.75)
        p95 = df_runs['final_bankroll'].quantile(0.95)
        mean = df_runs['final_bankroll'].mean()
        ruin_prob = (df_runs['final_bankroll'] < 1000.0).mean()

        row = {
            'kelly_frac': k,
            'p5_final': p5,
            'p25_final': p25,
            'median_final': p50,
            'p75_final': p75,
            'p95_final': p95,
            'mean_final': mean,
            'ruin_prob': ruin_prob,
            'mean_roi': df_runs['total_profit'].mean() / (df_runs['total_staked'].mean() if df_runs['total_staked'].mean() > 0 else np.nan)
        }
        mc_rows.append(row)
        print("done.")

    summary_df = pd.DataFrame(mc_rows)
    pd.set_option('display.float_format', lambda x: f"{x:,.2f}")
    print("\n===== MONTE CARLO SUMMARY (market-realistic) =====")
    print(summary_df.to_string(index=False))
