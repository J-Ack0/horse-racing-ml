# kaggle_v2 — race-conditional, leak-free UK+IRE win model

Everything here was run for real on the Pi; every number below is from the logs/CSVs in
this directory. Test set is always the **last 20% of dates**, split chronologically
(64/16/20 train/val/test by date, no race straddles a boundary, no shuffling across time).

## Files

| file | what it does |
|---|---|
| `features.py` | builds the engineered matrix from `data_ext/raceform.db` → `cache/features.pkl` |
| `common.py` | split logic, race-level metrics, race-softmax and Plackett-Luce custom objectives |
| `exp_objectives.py` | objective comparison at fixed inherited hyperparameters |
| `exp_rank_retry.py` | fair retry of XGBoost's built-in LambdaMART objectives |
| `exp_tune.py` | random search + rolling-origin (walk-forward) CV |
| `exp_final.py` | tuned final models, Plackett-Luce, blending, calibration |
| `exp_ablation.py` | warm-up-history ablation |
| `run_all.sh`, `summarise.py` | driver / results printer |

## Headline numbers (test = last 20% of dates)

UK+IRE trained, test window 2025-11-29 → 2026-05-27:

| model | AUC | AP | top-1 | MRR | NDCG@3 | race-logloss | ECE |
|---|---|---|---|---|---|---|---|
| session baseline (~28 feats, binary) | 0.6950 | 0.2250 | – | – | – | – | – |
| binary, median impute (untuned) | 0.7369 | 0.2636 | 0.2719 | 0.4827 | 0.4600 | 1.9703 | 0.0021 |
| binary, native NaN (untuned) | 0.7373 | 0.2636 | 0.2719 | 0.4828 | 0.4606 | 1.9701 | 0.0044 |
| race-softmax (untuned) | 0.7123 | 0.2314 | 0.2750 | 0.4854 | 0.4635 | 1.9638 | 0.0039 |
| binary, tuned | 0.7400 | 0.2650 | 0.2682 | 0.4807 | 0.4581 | 1.9699 | 0.0027 |
| race-softmax, tuned | 0.7428 | 0.2695 | 0.2729 | 0.4849 | 0.4633 | 1.9627 | 0.0038 |
| **Plackett-Luce top-3, tuned** | **0.7446** | **0.2740** | **0.2766** | **0.4870** | **0.4656** | **1.9572** | 0.0049 |
| blend(binary+softmax+PL) | 0.7445 | 0.2725 | 0.2726 | 0.4852 | 0.4630 | 1.9575 | 0.0030 |

Irish-only (own chronological split; test 2025-05-15 → 2025-10-14):

| model | AUC | AP | top-1 | MRR |
|---|---|---|---|---|
| legacy Irish-only baseline | 0.7340 | – | – | – |
| tonight's Irish-only baseline | 0.7159 | 0.2138 | – | – |
| binary, tuned | 0.7620 | 0.2593 | 0.2604 | 0.4682 |
| Plackett-Luce top-3 | 0.7643 | 0.2626 | 0.2756 | 0.4807 |
| blend(all three) | **0.7663** | **0.2647** | 0.2756 | 0.4791 |

## What actually helped

1. **Feature engineering (+0.042 AUC)** — by far the biggest single factor, and it is
   *not* the thing the brief asked for. 135 strictly point-in-time features vs ~28.
2. **Race-conditional objectives (+0.005 AUC, +0.009 AP, +0.008 top-1, −0.013 logloss)** —
   real but modest. Plackett-Luce top-3 > race softmax > binary on *every* race-level
   metric. The theory is vindicated in direction, not in magnitude.
3. **Hyperparameter search (+0.003 AUC binary, +0.031 AUC softmax)** — mostly it rescued
   the softmax model, which was badly undertrained at the inherited settings.
4. **Walk-forward CV** changed which configuration won vs the single validation split —
   so it mattered for *selection* even though it is not itself a modelling gain.

## What did NOT help

* **XGBoost's built-in `rank:pairwise` / `rank:ndcg` / `rank:map`** — worse than plain
  binary on everything (best retry: 0.7115 AUC). Purpose-built race likelihoods beat
  generic LambdaMART here.
* **Native NaN vs median imputation** — +0.0004 AUC. Noise.
* **Isotonic recalibration** — the model was already well calibrated (ECE 0.003–0.005);
  isotonic made it very slightly worse.
* **Blending** — matched but did not beat Plackett-Luce alone on UK+IRE.
* **Extra warm-up history (2021→2023 priming of the rolling stats)** — 0.7367 vs 0.7369
  AUC. No effect.

## Two data-integrity problems found and fixed here

* **Rows are stored in finishing order.** 58% of races have the winner as their first
  row. Any within-race ranking metric that breaks ties by row order therefore reads the
  answer off the row index. This inflated an early isotonic result to top-1 = 0.297;
  with rows shuffled within race (fixed seed) it is 0.272. `common.load()` now always
  shuffles. `results_objectives.csv` was produced *before* this fix — its AUC/AP are
  fine, but its top-1/MRR/NDCG for the heavily-tied `rank:ndcg`/`rank:map` rows are
  overstated; use `results_rank_retry.csv` instead.
* **Irish data ends 2025-10-14.** The UK+IRE test window contains zero Irish runners, so
  that model is trained on UK+IRE and tested on UK only, and the Irish-only model has to
  use an earlier test window. The two tables above are therefore not on the same dates.

## Leakage audit

`prize` was also found to be post-race (money won in *this* race) and is excluded.
All entity statistics are cumulative **up to but excluding the current race date**, which
is stricter than `cumsum() - self` (that variant leaks same-day and same-race results for
trainers/sires with multiple runners).
