# Horse Racing ML: Project Notes

**Last updated:** 2026-09-18
**Single source of truth.** This file replaces `FIXES.md`, `TRIAL_RUN_AUDIT.md`,
`docs/2026-09-14_live_data_pipeline_plan.md` and `ml/kaggle_v2/README.md`
(all folded in below, none of their facts dropped). The root `README.md` is a
short landing page that links here.

## Contents

1. [Overview and status](#1-overview-and-status)
2. [Data](#2-data)
   - [2.1 Data window decision](#21-data-window-decision)
   - [2.2 Sources and files](#22-sources-and-files)
   - [2.3 Naming conventions that break joins](#23-naming-conventions-that-break-joins)
3. [Model (kaggle_v2)](#3-model-kaggle_v2)
4. [Live data pipeline (The Racing API)](#4-live-data-pipeline-the-racing-api)
5. [Inference](#5-inference)
6. [Scoring and results log](#6-scoring-and-results-log)
7. [Analysis: course, UK vs Ireland, field size](#7-analysis-course-uk-vs-ireland-field-size)
8. [Testing](#8-testing)
9. [Legacy system (v1) and scraper audit](#9-legacy-system-v1-and-scraper-audit)
10. [Known issues and open TODOs](#10-known-issues-and-open-todos)
11. [Reproduce and verify](#11-reproduce-and-verify)
12. [Changelog](#12-changelog)

---

## 1. Overview and status

UK+Ireland horse-racing win-prediction project on a Raspberry Pi 5 (8 GB, no swap,
Arch Linux ARM). Two generations of code:

- **v1 (legacy, 2025):** XGBoost on ~28 features, Racing Post scrapers, Kelly
  betting filters. See [section 9](#9-legacy-system-v1-and-scraper-audit). The
  scrapers are broken and superseded by the API.
- **v2 (`ml/kaggle_v2/`, current):** leak-free 135-feature matrix, race-conditional
  objectives (binary, race-softmax, Plackett-Luce top-3), blended. Live data from
  The Racing API, live inference and scoring built 2026-09-16 to 2026-09-18.

Status as of 2026-09-18:

| Area | State |
|---|---|
| Model | Trained, `blend_all` AUC 0.7445 (UK+IRE test) / 0.7663 (Irish-only test) |
| Live data (racecards) | Working on the Free plan, live-verified 2026-09-16 |
| Live data (history backfill) | Plan upgraded to Standard (2026-09-20); 2026-05-28 to 2026-09-19 loaded into `live_extension.db` from exported JSON, **not yet used by inference/training** (see 4.6) |
| Inference | Working, ~3m50s per day, 444-529 runners |
| Scoring | Working (`score_predictions.py`); out-of-sample walk-forward over 115 days / 4,438 races: 27.9% top-1, 59.5% top-3 ([6.5](#65-walk-forward-backtest-on-the-post-may-races-2026-09-20)) |
| Tests | `data_collection/tests`: 50 pass, 4 live tests skipped by default; `ml/kaggle_v2/tests`: 8 pass |
| Data window | 2015-01-01 to 2026-05-27 (fixed, ends 2026-05-27; the gap to today grows daily) |

Repo pieces that matter (paths relative to repo root):

| Path | Purpose |
|---|---|
| `data_ext/raceform.db` | Historical table `data`, 2015-01-01 to 2026-05-27 (untracked) |
| `data/live_extension.db` | Live racecard rows, same schema plus `fetched_at` (untracked) |
| `data_collection/` | Racing API client, daily fetch, backfill, systemd units, tests; legacy `scrape_*.py` |
| `ml/kaggle_v2/` | v2 features, objectives, experiments, `inference.py`, `score_predictions.py`, cache, tests |
| `docs/PROJECT_NOTES.md` | This file |

---

## 2. Data

### 2.1 Data window decision

**Decision (2026-09-18): the modelling data window is 2015-01-01 to 2026-05-27,
from `data_ext/raceform.db`.** One file, 1,851,285 real rows (a raw `COUNT(*)` gives
1,851,286 because the table contains one stray header row whose `date` value is the
literal string `'date'`; always filter `date != 'date'`), consistent schema, 37
columns. `features.py` trains on 2023-05-27 onward with a 2021-01-01 warm-up.

**Not "to today".** The data ends 2026-05-27. The gap to the prediction date was 112
days on 2026-09-16, 113 on the 17th and 114 on the 18th, and grows daily until a
Standard Racing API plan allows a backfill.

**Verified 2026-09-18 across all three files** (commands in
[section 11](#11-reproduce-and-verify)):

| File | Rows | Range | Months |
|---|---|---|---|
| `data_ext/1988-2004.db` | 1,653,834 | 1988-01-01 to 2004-12-31 | 204 |
| `data_ext/2005-2014.db` | 1,555,126 | 2005-01-01 to 2014-12-31 | 120 |
| `data_ext/raceform.db` | 1,851,285 | 2015-01-01 to 2026-05-27 | 137 |

- Identical 37-column schema in all three, and no `race_id` appears in more than
  one file.
- The whole span 1988-01-01 to 2026-05-27 is **461 months with none empty**.
- The only gaps longer than 10 days between racing days: **Nov 1995 (14 days),
  Oct to Nov 1996 (12 days), Jan to Feb 2012 (18 days)**. The 2012 one is very likely
  the cold-snap abandonments; the two 1990s ones are unverified.
- Rating fields are sparse early: RPR populated on **4% of runners in 1988, 56% in
  1993**, roughly 80%+ from 1995; TS 0% (1988) rising to 50-70% in the 1990s;
  official rating (`or`) patchy until about 2000 (25% in 1988, 35-41% 1989 to 1997,
  56-73% 1998 to 2004). `sp` is 93-100% and `sire` ~100% throughout.
- Row count roughly doubles in 2003 (84K rows in 2002, 160K in 2003). **Cause not
  investigated** (coverage change, possibly added racing).
- 2020 is low (121,683 rows) because of COVID.
- From 2005 the fields are stable: 2005 to 2014 RPR 86-92%, TS 58-71%, OR 53-65%
  (the 2026-09-18 brief quoted TS 62-71% and OR 53-62%; the per-year table gives the
  wider ranges). In 2015 to 2026 RPR is 78-92%, TS 61-73%, OR 55-75% (2026 is partial
  year to 05-27).

**Recommendation:** use 2015 onward. Using 1995 or later is defensible for a longer
history but mixes eras with different coverage; pre-1995 is not recommended. The older
files are usable only for horse-career features (prior runs), where the sparse
ratings would not matter.

### 2.2 Sources and files

- `data_ext/` (untracked, large): `1988-2004.db`, `2005-2014.db`, `raceform.db`,
  `BHA_Full_ratings.csv`, and [`Kaggle_ReadMe.md`](../data_ext/Kaggle_ReadMe.md)
  (provenance of the static Kaggle export; form 2015 to present, archives 1988 to
  2014, author "Derek"). Not part of the tracked repo, so this link only resolves
  locally.
- Table `data` columns (37): `date, course, race_id, off, race_name, type, class,
  pattern, rating_band, age_band, sex_rest, dist, going, ran, num, pos, draw, ovr_btn,
  btn, horse, age, sex, wgt, hg, time, sp, jockey, trainer, prize, or, rpr, ts, sire,
  dam, damsire, owner, comment`. `or` is a reserved word: write `[or]` in SQL.
- Indexes added 2026-09-16 to `raceform.db` (file grew 765 MB to 1010 MB): `date`,
  `horse`, `jockey`, `trainer`, `sire`, `dam`, `damsire`. Needed for entity-scoped
  inference ([section 5](#5-inference)).
- `data/live_extension.db` is a **separate** file from the static export (which is
  never mutated). Same `data` schema plus `fetched_at`; upserts on
  `(date, race_id, horse)`.
- Path note: in this worktree the historical DB is `data_ext/raceform.db` (the path
  `features.py` uses); an earlier look at the main checkout found it as `data/raceform.db`.

### 2.3 Naming conventions that break joins

These all fail silently (empty features, not errors):

1. **Horse names carry a region suffix in history**: every horse is `"Name (IRE)"`
   or `"Name (GB)"` etc., no exceptions (even GB-bred get `(GB)`); the live API gives
   the bare name. Fixed for `horse` via the API's per-runner `region` field
   (`with_region_suffix()`). **Still open for `sire`/`dam`/`damsire`** (same suffix
   convention, but the Free racecard has no `sire_region`/`dam_region`/`damsire_region`).
2. **Irish course names lost `(IRE)` from 2025-10-15.** Before that date: `Naas (IRE)`,
   `Clonmel (IRE)`, `Downpatrick (IRE)`, `Dundalk (AW) (IRE)`. After: bare `Naas`,
   `Clonmel`, `Downpatrick`, `Dundalk (AW)` (Clonmel bare from 2025-10-23, Naas from
   2025-11-09, Downpatrick from 2026-03-29, Dundalk (AW) from 2025-10-17). Irish
   history in `raceform.db`: 338,817 rows ending 2025-10-14 under `(IRE)` names;
   all other course names (UK plus the bare-named Irish rows) 1,512,468 rows to
   2026-05-27. `features.py` derives `is_ire` from the `(IRE)`
   suffix, so Irish runners after that date and every live Irish race get
   `is_ire = 0` ([section 7](#7-analysis-course-uk-vs-ireland-field-size)).
3. **Weight and distance formats**: history stores weight as stone-lbs (`"10-0"`) and
   distance as `"1m2f"`; the API gives plain lbs (`"140"`) and plain furlongs
   (`"10.0"`). Converted in the row mapper (`lbs_to_wgt_str`, `furlongs_to_dist_str`).
4. **Unrated official rating** comes back as the literal `"–"` (en dash), not null.
5. **`num`**: predictions store it as a float (`2.0`), the API as a string (`"2"`).
   Normalised in the scorer.
6. **`(AW)`** survives in both spellings, so `is_aw` is unaffected.

---

## 3. Model (kaggle_v2)

Race-conditional, leak-free UK+IRE win model. Every number below is from the
logs/CSVs in `ml/kaggle_v2/` (run on the Pi). The test set is always the **last 20%
of dates**, split chronologically (64/16/20 train/val/test by date, no race
straddles a boundary, no shuffling across time). Features: 135 in the base list
`cache/feature_names.txt` (includes `is_ire`; `wc -l` shows 134 because the file has no
trailing newline); the Irish-only model uses 134 (drops `is_ire`). Earlier notes and the
`inference.py` docstring said "134" for the base model; 135 is correct.

### 3.1 Files

| File | What it does |
|---|---|
| `features.py` | builds the engineered matrix from `data_ext/raceform.db` into `cache/features.pkl` |
| `common.py` | split logic, race-level metrics, race-softmax and Plackett-Luce custom objectives |
| `exp_objectives.py` | objective comparison at fixed inherited hyperparameters |
| `exp_rank_retry.py` | fair retry of XGBoost's built-in LambdaMART objectives |
| `exp_tune.py` | random search plus rolling-origin (walk-forward) CV |
| `exp_final.py` | tuned final models, Plackett-Luce, blending, calibration (`--ire` for Irish-only) |
| `exp_ablation.py` | warm-up-history ablation |
| `run_all.sh`, `summarise.py` | driver / results printer |
| `inference.py`, `score_predictions.py` | live scoring ([sections 5](#5-inference) and [6](#6-scoring-and-results-log)) |

Saved models in `cache/`: `final_{binary,softmax,pltop3}.json` (UK+IRE base, used
for live inference) and `final_{binary,softmax,pltop3}_ire.json` (Irish-only).

### 3.2 Headline numbers (test = last 20% of dates)

UK+IRE trained, test window 2025-11-29 to 2026-05-27:

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

Irish-only (own chronological split; test 2025-05-15 to 2025-10-14):

| model | AUC | AP | top-1 | MRR |
|---|---|---|---|---|
| legacy Irish-only baseline | 0.7340 | – | – | – |
| Irish-only baseline (that night) | 0.7159 | 0.2138 | – | – |
| binary, tuned | 0.7620 | 0.2593 | 0.2604 | 0.4682 |
| Plackett-Luce top-3 | 0.7643 | 0.2626 | 0.2756 | 0.4807 |
| blend(all three) | **0.7663** | **0.2647** | 0.2756 | 0.4791 |

The two tables are on different date windows, so they are not directly comparable.
The **deployed** live model is the UK+IRE base blend (0.7445), not the Irish-only
0.7663. The blend is the geometric mean of the per-race-normalised probabilities of
binary, softmax and PL, renormalised per race (`exp_final.py::blend()`).

### 3.3 What actually helped

1. **Feature engineering (+0.042 AUC)**: by far the biggest factor, and not what the
   brief asked for. 135 strictly point-in-time features vs ~28.
2. **Race-conditional objectives (+0.005 AUC, +0.009 AP, +0.008 top-1, -0.013
   logloss)**: real but modest. Plackett-Luce top-3 > race softmax > binary on every
   race-level metric.
3. **Hyperparameter search (+0.003 AUC binary, +0.031 AUC softmax)**: mostly rescued
   the softmax model, badly undertrained at the inherited settings.
4. **Walk-forward CV** changed which configuration won vs the single validation
   split, so it mattered for selection though it is not itself a modelling gain.

### 3.4 What did NOT help

- XGBoost's built-in `rank:pairwise` / `rank:ndcg` / `rank:map`: worse than plain
  binary on everything (best retry 0.7115 AUC).
- Native NaN vs median imputation: +0.0004 AUC, noise.
- Isotonic recalibration: model already well calibrated (ECE 0.003-0.005); isotonic
  made it very slightly worse.
- Blending: matched but did not beat Plackett-Luce alone on UK+IRE.
- Extra warm-up history (2021 to 2023 priming): 0.7367 vs 0.7369 AUC, no effect.

### 3.5 Data-integrity problems and leakage rules

- **Rows are stored in finishing order.** 58% of races have the winner as their first
  row, so any within-race ranking metric that breaks ties by row order reads the answer
  off the row index. This inflated an early isotonic result to top-1 = 0.297 (0.272
  after shuffling within race with a fixed seed). `common.load()` now always shuffles.
  `results_objectives.csv` predates this fix: its AUC/AP are fine but top-1/MRR/NDCG for
  the heavily tied `rank:ndcg`/`rank:map` rows are overstated; use
  `results_rank_retry.csv`.
- **The README's earlier claim "Irish data ends 2025-10-14" was wrong.** It is a
  labelling artifact of the `(IRE)` suffix change ([2.3](#23-naming-conventions-that-break-joins)).
  The UK+IRE test window (2025-11-29 to 2026-05-27) contained **13,818 Irish runners in
  1,200 races** (of 78,837 runners / 8,073 races) labelled `is_ire = 0`, so the 0.7445
  AUC was on a mixed set, not UK only. The Irish-only model therefore only saw data up
  to 2025-10-14.
- **Leakage audit:** never use current-row `rpr`, `ts`, `sp`, `pos`, `btn`, `ovr_btn`,
  `time`, `prize`, `comment`. `prize` was found to be post-race (money won in this
  race) and is excluded. `or` (official rating) is pre-race and is used. All entity
  statistics are cumulative **up to but excluding the current race date** (day
  granularity), stricter than `cumsum() - self`, which leaks same-day and same-race
  results for trainers/sires with several runners. Raw horse/jockey/trainer/sire/dam
  names are group keys only, never features.
- `features.py::build()` sorts by `(date, race_id)`; dead-heat handling keeps races with
  at least one winner. Changed 2026-09-16: races where **every** row is unresolved (a
  future/today racecard) are now kept instead of dropped; historical behaviour is
  unchanged. A defragmenting `df.copy()` was added after the `day_stats` loop
  (performance only).

### 3.6 Feature importance (2026-09-18)

`is_ire` carries 0.01-0.04% of gain (rank ~118-122 of 126-131); `is_aw` 0.05-0.11%;
`tc_wr` ~0.7-0.9%, `jc_wr` ~0.4-0.5%. The models lean on the horse's own form: the
top features are `prior_rpr_rk` (9.0% in binary), `field_size` (8.5%), `h_rel_ema_rk`,
`prior_rpr_z` (16-18% in softmax and PL), `h_rel_ema_z`, `h_rel_ema`,
`prior_rpr_best3_z`, `trn_wr_z`.

---

## 4. Live data pipeline (The Racing API)

### 4.1 Decisions

1. **Source: The Racing API (theracingapi.com)**, not the Racing Post scraper. The
   scraper is confirmed broken ([section 9](#9-legacy-system-v1-and-scraper-audit)).
   The API gives stable UK+Ireland+HK coverage; today's racecard updates every 3
   minutes, tomorrow's every 15. No racing API was documented anywhere in the repo
   before 2026-09-14: this was a fresh decision, not a resumption.
2. **Credentials** live in a gitignored repo-root `.env` as `USERNAME` / `PASSWORD`
   (the first plan said `RACING_API_USERNAME`/`RACING_API_PASSWORD`; superseded).
   `.env` is not copied into git worktrees, so the client also loads it from the main
   repo root via `git rev-parse --git-common-dir`. Because `USERNAME`/`PASSWORD` are
   generic names, values already in the real environment win (`load_dotenv` does not
   override); check `env | grep -E '^(USERNAME|PASSWORD)='` if auth looks wrong.
3. **Two-script split, deliberately, to protect the point-in-time rule:**
   `fetch_daily_racecards.py` writes pre-race fields only (no
   pos/sp/rpr/ts/prize/comment); `backfill_history.py` writes post-race fields for
   training history only. Keeping them separate makes it structurally impossible to
   leak a result into the same day's pre-race features.
4. **Separate DB**: writes go to `data/live_extension.db`, never into the static export.

### 4.2 Code (`data_collection/`)

- `racingapi_client.py`: HTTP Basic client; class-level shared rate-limit clock;
  `racecards_free()`, `racecards_standard()`, `results_today_free()`, `results()` and
  `results_all_pages()` (paginates `skip += 500` until `total`); `RacingAPIPlanError`
  for "Standard Plan required"; `--probe` mode. Loops `REGIONS = ["gb", "ire"]`
  because `region_codes` takes one region per call.
- `live_db.py`: SQLite writer, upsert on `(date, race_id, horse)`.
- `fetch_daily_racecards.py`: daily pre-race fetch; `--day today|tomorrow`. Intended
  runs: evening (tomorrow's card) and pre-race (today's card, to catch declaration
  changes).
- `backfill_history.py`: pulls settled results from 2026-05-28 (dataset end + 1) to
  yesterday in one ranged call with pagination; idempotent; defaults self-heal a day
  that failed. Prints a plan-required message and exits 1 on the Free plan.
- `systemd/`: `racingapi-daily-fetch.{service,timer}` (01:00: backfill yesterday's results, fetch today's racecard,
  then `fast_inference.py`; the backfill step is non-fatal) and `racingapi-daily-score.{service,timer}` (23:00: score today
  against `/results/today/free`; `Persistent=false` because the Free plan only serves
  today's results). Both call `data_collection/run_daily.sh fetch|score`. **Installed
  and enabled 2026-09-19** as symlinks in `~/.config/systemd/user/` (`systemctl --user
  link` the `.service` files as well as enabling the `.timer` files), with
  `loginctl enable-linger` on so they run without a desktop session. Systemd user
  timers rather than cron (no crontab on this Pi). The units point at the worktree
  path (`.claude/worktrees/quirky-wobbling-teapot`), because `inference.py`/the scorer
  and the `venv/` exist only there; re-link them if that worktree moves or the branch
  is merged. The fetch unit retries up to 3 times, 10 min apart, in case inference is
  OOM-killed. Each run pushes a phone note through `notify-phone.sh` (`[Done]` with race/runner counts and the first 12 top picks after the 01:00 fetch+inference; `[Done]` with top-1/top-3/precision@3 after the 23:00 score; `[Error]` with the log tail on any failure). An offline phone queues the note and never fails the job. `score_predictions.py` now also reports precision@3 (added 2026-09-19, with a test). The older `racingapi-evening-fetch` (18:00) and
  `racingapi-morning-backfill` (07:00) units are still **not installed** (backfill
  needs a Standard plan).

### 4.3 Free vs Standard plan (live-verified 2026-09-16)

The table below is the Free-plan state verified 2026-09-16. **The account moved to the Standard plan on 2026-09-20**; the Standard endpoints have not been probed live from this repo yet (the history loaded in 4.6 came from exported files).

| Endpoint | Free plan | Notes |
|---|---|---|
| `/v1/racecards/free` | works | carries every pre-race field the model needs |
| `/v1/racecards/standard` | 401 `Standard Plan required` | adds bookmaker odds etc. |
| `/v1/results/today/free` | works, today only | positions, no sp/rpr/ts/prize/comment/btn/time |
| `/v1/results` (historical) | 401 `Standard Plan required` | needed for `backfill_history.py` |

- **Rate limit** is 1 request/second on Free (confirmed by a live 429, "Rate limit
  exceeded: 1 per 1 second"); Standard 5/s; Pro variable.
- `region_codes` takes exactly one region per call; `"gb+ire"` returns 422
  ("unrecognised region code").
- Free racecard runner fields include `ofr` (official rating), `lbs`, `draw`, `sire`,
  `dam`, `damsire`, `headgear`, `jockey`, `trainer`, `owner`, `horse_id`, `region`,
  `form`, `last_run`; race-level `race_class`, `going`, `age_band`, `sex_restriction`,
  `pattern`, `rating_band`, `distance_f`, `field_size`, `off_time`. There is no
  `sire_region`/`dam_region`/`damsire_region` and no distance string like `"6f210y"`.
- **Free racecards keep withdrawn horses** (non-runners).
- Field-name inconsistency between endpoints: racecards use `ofr` and `lbs`/`number`;
  free results use `or`, `weight`/`weight_lbs`, `number`, `position`, and race keys
  `off`, `dist_f`, `class`, `sex_rest`. `results_to_rows()` tries `or` then `ofr`.
- `/results` (paid) field mapping is **unverified** against a real paid response;
  assumed: `position`->pos, `sp_dec`->sp, `btn`, `weight_lbs`->wgt,
  `performance_rating`->rpr, `speed_rating`->ts, `comments`->comment, `prize`,
  `ofr`->or, runners nested under races. Re-verify immediately after upgrading.
- `racecards_free(when="tomorrow")` is untested live (only "today" verified).

### 4.6 History gap fill, 2026-09-20

The gap (2026-05-28 to yesterday, 115 days) was filled from two exported `/v1/results` JSON
files sent by Taildrop (`2026-05-26_2026-06-27_results.json`, `2026-06-26_2026-09-20_results.json`,
kept in `~/Desktop/taildrop-inbox/`, not in git). Together they cover every day 05-28 to 09-19
with no missing day; the 153 overlapping races are identical.

- **Loaded**: `data_collection/load_results_json.py FILE...` (idempotent, backs up
  `live_extension.db` first, `--dry-run`, `--regions GB,IRE,FR|all`, `--start`). Loaded GB+IRE+FR
  only (5,675 of 5,987 races, 53,442 runner rows over 115 days); the other regions (HK, USA, ARG,
  ...) were skipped, rerun with `--regions all` to add them. `live_extension.db` now holds 53,875
  rows over 116 days (05-28 to 09-20; the 20th is the pre-race card).
- **Standard-plan endpoint facts (live-verified 2026-09-20)**: `/v1/results` takes ONE region per call (`gb,ire` is rejected) and `limit <= 100` races per page (`total` counts races); its output for 2026-09-19 matched the exported files exactly (53 GB+IRE races, 581 runners). `/v1/racecards/standard` returns `odds[]` per runner (about 28 bookmakers, some with decimal `"SP"`). The racecards' `off_time` is a 12h clock without am/pm; `fetch_daily_racecards.py` now stores the 24h time from `off_dt` (before this fix live predictions had `off_hour` 2 instead of 14).
- **Mapping bugs found and fixed** by comparing 706 runners against `raceform.db` on 05-26/27
  (`data_collection/results_mapping.py`, now used by `backfill_history.py`; the old "assumed"
  mapping is gone): `performance_rating` and `speed_rating` are **not** raceform's `rpr` / `ts`
  (history `rpr` is blank where the vendor has 113; the vendor's own `rpr` and `tsr` fields are
  empty in every row), so `rpr` and `ts` are **NULL for all loaded rows**; `wgt` must come from the
  stone-lb `weight` ("8-13"), not `weight_lbs`; `off` comes from `off_dt` (24h), not the 12h `off`;
  `dist` drops the yardage ("2m13y" -> "2m"); `time` pads seconds ("4:5.30" -> "4:05.30"); `sp` stays
  fractional text like history. Exact match on pos, prize, draw, going, off; 99% on wgt/btn/sp, 96%
  on official rating.
- **Merge days**: the 16th's racecard rows held bare horse names (pre-suffix-fix), so the loader
  matches on the bare name and renames to the suffixed form.
- **Used by inference since 2026-09-20**: `fast_inference.py` reads `live_extension.db` as history for
  every later day (see [5.0](#50-fast-path-and-two-feature-bugs-fixed-2026-09-20)); training still reads
  only the static export. `rpr`/`ts` are NULL in the gap, so `prior_rpr`/`prior_ts` do not see those runs.
- `raceform.db` contains French races (about 10% of May's rows); FR in the gap is loaded for the same
  reason.
- Correction to an earlier note in this session: race_ids from the results and racecard endpoints do
  match on recent days (both `rac_3229...`); no id-scheme problem.

### 4.4 Vendor documentation (found 2026-09-15)

`api.theracingapi.com/documentation` is gated behind signup, but Context7's public
index (`context7.com/websites/api_theracingapi`, `/llms.txt`) mirrors it. Base URL
`https://api.theracingapi.com`. Other endpoints not yet used: `/v1/{horses,jockeys,trainers,sires,dams,owners}/search`,
`/v1/horses/{id}/standard`, `/v1/horses/{id}/analysis/distance-times`,
`/v1/trainers/{id}/analysis/{courses,jockeys,horse-ages}`,
`/v1/sires/{id}/analysis/classes`, `/v1/damsires/{id}/analysis/classes`,
`/v1/meets/free` (North America). Standard racecard extras not yet consumed:
`trainer_14_days`, `odds[]` (bookmaker quotes), `silk_url`, `wind_surgery`, `colour`,
`dob`, `breeder`.

### 4.5 Bugs found and fixed in the pipeline (2026-09-16)

- **Per-instance rate-limit clock**: two client objects in one process each thought
  they had a fresh 1 req/s allowance. Now class-level, with a regression test.
- **`ofr` vs `or`** field-name inconsistency (caught by the tests).
- **Weight/distance formats** and **horse-name suffix**: see [2.3](#23-naming-conventions-that-break-joins).
- **Test bugs** (tests too strict): asserting `__dict__` contents rather than repr;
  requiring non-empty `pattern`/`sex_rest` when an empty string is legitimate.

---

## 5. Inference

### 5.0 Fast path and two feature bugs fixed (2026-09-20)

**Use `ml/kaggle_v2/fast_inference.py --date YYYY-MM-DD`** (about 12 to 20 s per day, under 1.5 GB).
It builds features only for the day's runners: each horse's own past runs plus SQL running
counts per jockey/trainer/sire/dam/damsire/pair strictly before the date, passed to
`features.build(..., entity_counts=...)`. History = `raceform.db` (to 2026-05-27) plus every
finished day in `live_extension.db` dated before the target, so predicting D+1 automatically sees D's
results (walk-forward). **Verified identical to the training matrix**: 0 of 135 features differ on
2026-05-20, 2025-10-04 and 2024-06-15 (`ml/kaggle_v2/verify_fast_vs_training.py`).

Two bugs in the older `inference.py` (still there; it now scores in memory-bounded chunks of whole
races and passes the fix below) made every earlier live prediction use different features from
training:

1. **`build()` dropped a third of the history.** Its "race with a result but no winner = data
   error" filter ran on history loaded per entity, where a past race usually has only some rows
   loaded and its winner is often absent. On 2026-09-06 32% of loaded history rows were dropped
   (for example a horse with 26 real prior runs kept 14), understating `h_runs_prior`,
   `days_since_run`, `jky_runs` and friends. The full table has 0 such races. Fix:
   `build(..., drop_zero_winner_races=False)` for inference (default unchanged for training). The
   earlier claim below that entity stats "depend only on the entity's own rows" was wrong because of
   this filter, and it is also why scoring a day in race chunks looked inexact (up to 0.086 in
   probability); after the fix chunked and whole-day runs are identical.
2. **History window mismatch.** Training built features from `HISTORY_START=2021-01-01`
   (`features.py` default) while inference loaded everything back to 2015, inflating every entity
   count. `fast_inference.py` uses 2021-01-01.

The live results logged for 2026-09-16 to 19 in [6.3](#63-results-by-day-1-pick-per-race)
(about 21% top-1) were produced with bug 1 and are superseded by the walk-forward in
[6.5](#65-walk-forward-backtest-on-the-post-may-races-2026-09-20).

**Non-runners (2026-09-20)**: the racecard APIs keep a withdrawn horse in its race (Standard racecards mark it `number: "NR"` and still quote prices); 13 of 187 runners on 2026-09-20 were NRs across 11 races (one race fell from 5 to 3 runners). `fast_inference.py` now drops them for today (`drop_non_runners`, uses `racecards_standard`, resets `ran`) and the odds report skips them. Withdrawals happen during the morning, so the 01:00 predictions can be stale; rerun `fast_inference.py` / `today_odds_report.py` shortly before racing.

Limits: gap rows loaded from the results export have `rpr`/`ts` NULL (the vendor's
performance/speed ratings are a different scale), so `prior_rpr`/`prior_ts` do not see those runs.
Nightly results come from `backfill_history.py` (default start = day after the last day with results;
first live use on the Standard plan is untested).

### 5.1 Older entity-scoped path (`inference.py`)


`ml/kaggle_v2/inference.py --date YYYY-MM-DD` scores a day's racecard with the base
(UK+IRE, includes `is_ire`) blend_all model. Output:
`ml/kaggle_v2/predictions/predictions_<date>.csv` (gitignored run artifact) plus a
top-pick-per-race console summary. Columns: `date, race_id, course, off, race_name,
horse, jockey, trainer, num, binary, softmax, pl_top3, blend_all, rank_in_race`.

Pipeline: (1) load today's pre-race rows from `live_extension.db`; (2) load **only
the historical rows that can affect today's field**: for each of
horse/jockey/trainer/sire/dam/damsire, every past row matching a value running today
(six indexed queries, unioned and de-duplicated on `(date, race_id, horse)`); (3)
concatenate and run `features.build()`; (4) keep today's rows, select the 135 base
features, run the three boosters, blend by geometric mean and renormalise per race.

**Why entity-scoped**: every rolling/entity stat is computed per entity, so an
entity's value depends only on its own rows. The first working run used the full
table (`HISTORY_START=2021-01-01`, ~900K rows) and took **~22 minutes**. Scoped loading
brought it to **~780K rows and ~3m50s** (2026-09-17, 444 runners); the 18th loaded
967,487 rows for 529 runners (528 horses, 231 jockeys, 251 trainers, 233 sires, 519
dams, 250 damsires). Remaining time is `build()`'s groupby/cumsum work.

**Memory**: the run reaches ~2.7 GB RSS (fragmented frame); on 2026-09-17 one attempt
was killed by the OOM killer (8 GB, no swap) and succeeded on retry once memory
recovered. Check `free -h` first.

**Known data gap**: history ends 2026-05-27, so recent-form features
(`days_since_run`, entity `day_stats`, `prior_*`) understate anything since then.
`inference.py` prints the gap size on every run. Predictions are directional, not
calibrated.

**Bugs found only by running live** (all fixed; 4 with regression tests in
`data_collection/tests/test_fetch_daily_racecards.py`):

1. `build()` dropped every race with zero winners so far, silently deleting all of
   today's rows.
2. Weight/distance in plain lbs/furlongs were parsed to NaN.
3. **The big one**: bare API horse names did not match suffixed history names, so
   every `prior_*`/`h_*` feature was NaN for every live runner (30 all-NaN features on
   the first correct run). After the fix only 1 feature is all-NaN (`sire_wr_z`, from
   the open sire/dam/damsire suffix gap).
4. The `num` float/string mismatch in the scorer, and the racecard's withdrawn horses
   ([section 6](#6-scoring-and-results-log)).

---

## 6. Scoring and results log

### 6.1 Metric semantics

`ml/kaggle_v2/score_predictions.py --date YYYY-MM-DD` scores a day against
`/v1/results/today/free`, so it only works for **today** on the Free plan.

- **top1_accuracy**: the model's #1 pick actually won.
- **top3_accuracy**: the #1 pick finished in the top 3.
- **precision@3**: of the model's top 3 picks per race, the share inside the actual top 3.
- **mean_position_error**: mean |predicted rank - actual finishing position| across
  finishers (PU/F/UR etc. excluded from this average; counted as "did not win" in
  top-1/top-3). Lower is better; 0 is perfect. Random ordering is about 3.5.
- Joins on `(race_id, num)`, both normalised to integer strings. Each race is
  re-ranked among horses that actually ran (the free racecard keeps withdrawn horses);
  `n_nonrunners_dropped` is reported. Unit-tested on the pure `score()` function.

### 6.2 Free historical results source

`https://www.horseracing.net/results/<course>/<dd-mm-yy>` (e.g.
`/results/yarmouth/16-09-26`) is server-rendered and returns a full card with
1st/2nd/3rd per race via WebFetch. Racing Post, Sporting Life, AtTheRaces, RacingTV
and BBC results pages do not work for a fetcher (JS apps or blocked; Racing Post
day-index pages redirect to today, individual race URLs work only when the exact race
id is known). This lets past days be scored (top 3 only, no full order, no
sp/rpr/ts), but it cannot substitute for `backfill_history.py`.

### 6.3 Results by day (#1 pick per race)

**Superseded**: the rows below were scored with the history-dropping feature bug ([5.0](#50-fast-path-and-two-feature-bugs-fixed-2026-09-20)). The corrected out-of-sample numbers are in [6.5](#65-walk-forward-backtest-on-the-post-may-races-2026-09-20).

| Day | Races | Top-1 | #1 pick in top 3 | Precision@3 | Mean position error |
|---|---|---|---|---|---|
| 2026-09-16 | 34 | 11/34 = 32.4% | 22/34 = 64.7% | 50/102 = 49.0% | n/a (top-3 only) |
| 2026-09-17 | 39 | 5/39 = 12.8% | 15/39 = 38.5% | 37/117 = 31.6% | n/a (top-3 only) |
| 2026-09-18 | 44 | 10/44 = 22.7% | 22/44 = 50.0% | 60/132 = 45.5% | 3.36 (random about 3.5) |
| 16th + 17th | 73 | 16/73 = 21.9% | 37/73 = 50.7% | 87/219 = 39.7% | |
| All three | 117 | 26 wins, about 22% | | | |
| 2026-09-19 | 53 | 10/53 = 18.9% | 23/53 = 43.4% | 70/159 = 44.0% | 3.55 (random about 3.5) |

- 18th detail: inference ran about 19:30 IST after most races had finished, so it is a
  post-hoc run (no leakage, not a prospective test). `/results/today/free` returned 42
  finished races (416 runners, full positions incl. PU/F); a later re-run at 44 of 47
  races gave the table row above (the first pass at 42 races: 23.8% / 52.4% / 45.2% /
  mean position error 3.34). 6 of the first 42 races had a withdrawn #1 pick; 51-52
  declared runners never ran. Un-reranked, the scorer gave 9/36 = 25.0%, 19/36 = 52.8%
  (36 races) or 9/42 and 19/42 counting withdrawn picks as misses.
- Reference: training-time top-1 about 0.27, top-3 about 0.60; random top-1 at ~11
  runners is about 9%. Mean position error is only slightly better than random, so the
  model's value is at the top of the ranking, not the full order.
- The 16th ran on the degraded feature set (before the horse-name fix, ~25 features
  all-NaN); the 17th and 18th on the fixed set. The 17th was worse, but n=39 (top-1
  SE about 7 points) cannot separate the fix's effect from day-to-day variance.
- The 16th has 34 scored races (an earlier note said 35): one Kelso race was dropped
  by `build()` (3 of 327 runners).
- The results API returns positions but **no features**: no sp, rpr, ts, prize,
  comment, btn or time, so it can score predictions but cannot feed `prior_rpr`/`prior_ts`
  history.
- The evening of the 17th was never scored through the API (the "today" window
  closed at midnight); the 17th was scored via horseracing.net.

---

### 6.5 Walk-forward backtest on the post-May races (2026-09-20)

Every day from 2026-05-28 to 2026-09-19 predicted with `fast_inference.py` from history = raceform
plus all earlier days' results, day's own result columns blanked, then scored
(`ml/kaggle_v2/backtest_walkforward.py`, `evaluate_backtest.py`, `run_backtest.sh`; predictions and
`report.md` in `ml/kaggle_v2/predictions/backtest_wf/`, gitignored). These races are after the model's
training and held-out window, so this is a genuine out-of-sample test: **115 days, 4,438 GB+IRE
races, 41,381 runners**, 15 minutes end to end, no failures.

| Metric (#1 pick per race, blend_all) | Value | 95% CI | Held-out reference |
|---|---|---|---|
| top-1 (pick won) | 27.9% | 26.6 to 29.2 | 27.3% |
| top-3 (pick placed) | 59.5% | 58.1 to 60.9 | about 60% |
| precision@3 | 52.2% | 51.4 to 53.0 | |
| MRR of the winner | 0.490 | | about 0.49 |
| mean position error | 2.76 | | random about 3.5 |
| runner AUC (win) | 0.739 | | 0.7445 |
| average precision (win) | 0.275 | 0.258 to 0.294 | about 0.27 |

Per-day distribution (n=115): top-1 mean 27.9%, sd 7.5 (p5 16.5, p25 23.1, median 28.0, p75 33.3,
p95 40.4); top-3 mean 60.0%, sd 8.1 (p5 47.0, p95 73.4); precision@3 mean 52.6%, sd 5.4.

Runner-level threshold table (flag a runner if `blend_all` >= t; positive = wins):

| t | flagged per race | precision | recall | F1 | flat ROI at SP |
|---|---|---|---|---|---|
| 0.10 | 3.9 | 18.5% | 72.1% | 0.294 | -16.5% |
| 0.15 | 2.1 | 24.2% | 49.9% | **0.326** | -12.5% |
| 0.20 | 1.1 | 30.2% | 33.1% | 0.316 | -9.0% |
| 0.30 | 0.34 | 40.8% | 14.0% | 0.209 | -6.6% |
| 0.40 | 0.10 | 52.6% | 5.5% | 0.100 | -4.2% |
| 0.50 | 0.03 | 62.6% | 2.0% | 0.038 | -4.3% |

#1 pick only, by minimum confidence of the pick: p >= 0.20 covers 68% of races with 32.7% top-1 /
68.7% top-3; p >= 0.30 covers 30% with 41.0% / 78.4%; p >= 0.40 covers 10% with 52.0% / 87.2%.
Calibration by decile is close to the diagonal (top decile mean p 0.298, win rate 0.318).

Splits (top-1 / top-3): GB 28.5% / 61.1% (3,811 races), Ireland 24.2% / 50.1% (627); fields of 8 or
fewer 34.1% / 71.2%, 9-12 23.7% / 54.2%, 13-16 19.8% / 39.3%, 17+ 19.2% / 38.9%; Flat 28.2%,
Hurdle 25.5%, Chase 28.4%; by month 25.7 (May, 5 days), 27.7, 29.2, 27.4, 27.0.

Reading it: the model holds its held-out level out of sample, so the poor live numbers logged in 6.3
came from the feature bug in [5.0](#50-fast-path-and-two-feature-bugs-fixed-2026-09-20), not from the
model. It is still **not a betting edge**: flat ROI at SP for the #1 pick is -11.0% (all races) and
stays negative at every confidence threshold (-4% to -9%), in line with the -15.2% held-out figure in
6.4. Ireland is weaker than GB here (24% vs 29% top-1, 627 races).

### 6.6 What share of picks is at a bettable price (2026-09-20)

`ml/kaggle_v2/bettable_analysis.py` on the walk-forward backtest (4,438 races) joined to the
starting price. Break-even decimal odds for a confidence band = 1 / precision of all runners at or
above the threshold (t = 0.20 has precision 30.2%, so break-even 3.34 decimal, 7/3 fractional).
"Bettable" = SP at or above that.

| t | runners | precision | break-even | bettable share | win rate of the bettable ones | ROI of bettable | ROI of the rest |
|---|---|---|---|---|---|---|---|
| 0.15 | 9,086 | 24.1% | 4.16 | 50.4% | 13.0% | -16.5% | -8.4% |
| 0.20 | 4,817 | 30.0% | 3.34 | 50.1% | 18.2% | -9.7% | -8.3% |
| 0.30 | 1,491 | 40.6% | 2.46 | 50.7% | 26.6% | -6.4% | -6.8% |
| 0.40 | 453 | 52.3% | 1.91 | 42.6% | 36.3% | -2.4% | -5.5% |

About half of the picks clear the price bar at every threshold (#1 picks only: 49% to 53%), **but the
ones that clear it do not win at the threshold's precision** (t=0.20: 18.2% against the 30.2% needed),
because a price above break-even is where the market disagrees with the model, and the market is
right more often. #1 picks by SP band: model p vs market-implied vs actual is 0.392 / 0.630 / 0.595
at SP < 2, but 0.191 / 0.130 / 0.102 at SP 6-10 and 0.175 / 0.070 / 0.057 at SP 10+ (the model
over-rates long prices). Value rule p x SP - 1 >= 0 selects 39.6% of #1 picks (win rate 16.9%, ROI
-14.2%); even >= 0.20 edge (26% of picks) loses 15.8%. Nothing turned positive; the least bad
subsets are the high-confidence ones (-2% to -6%). Caveat: SP is the closing price, not necessarily
obtainable, and the best price across bookmakers earlier in the day can differ.

`ml/kaggle_v2/today_odds_report.py [--date]` lists the day's #1 picks with the best current
bookmaker price (racecards/standard, about 28 bookmakers per runner, "SP" entries skipped) or, for
finished races, the closing SP from `/v1/results`, with break-even odds, bettable flag, model edge and
the market's own implied probability (overround removed). First run 2026-09-20 12:23, before any race:
15 of 21 #1 picks (71%) at a bettable best price and 10 of 21 with model edge >= 0; the best of 28
prices is biased upward relative to the SP, so tonight's closing report is the fair comparison. The
01:00 and 23:00 jobs add these lines to the phone notes.

### 6.4 Betting math: break-even odds and holdout ROI at SP (2026-09-19)

Conventions: 1 unit flat stake on the model's #1 pick, decimal odds include the stake
(5/2 = 3.5), no commission or tax. Break-even decimal odds = 1 / hit rate. To target a
return r you need odds of (1 + r) / hit rate.

**Napkin numbers**

| Hit rate (#1 pick wins) | Source | Break-even odds | For +5% | For +10% |
|---|---|---|---|---|
| 27.3% | held-out test, blend_all (`results_final.csv`) | 3.67 (about 8/3) | 3.85 | 4.04 |
| 22.2% (26/117) | live, 16th to 18th | 4.50 (7/2) | 4.73 | 4.95 |
| 15.6% to 30.6% | live 95% interval | 6.4 down to 3.3 | - | - |

- The live interval is wide: 117 races give an ROI standard error of about 15 points
  (per-bet return SD is roughly 1.65 at these odds). Seeing a 2.5-point ROI standard
  error would take about 4,000 to 5,000 bets. Live results cannot confirm or reject a
  profit at this sample size.
- Kelly stake at odds 4.0: 3.0% of bankroll using the 27.3% hit rate; negative (no bet)
  using the live 22.2%.
- Losing runs: 72.7% of bets lose, so 10 losses in a row happens about 4% of the time
  from any starting point, and the longest losing run in 100 bets is typically about 10.

**Held-out test priced at real starting prices** (`ml/kaggle_v2/holdout_roi.py`;
window 2025-11-29 to 2026-05-27, 7,961 races with a complete SP book):

| Strategy | Bets | Win rate | Mean odds | ROI |
|---|---|---|---|---|
| Model #1 pick at SP | 7,961 | 27.3% | 4.66 | -15.2% |
| Market favourite at SP | 7,961 | 33.8% | 2.98 | -12.6% |
| Every runner at SP | 79,167 | 10.2% | 27.5 | -20.6% |

- The SP book sums to 121.2% per race (about 17.5% takeout), so blind betting loses
  12 to 20% before any edge.
- The model's winners paid 3.11 on average (from 27.3% x 3.11 = 0.85) against the 3.67
  needed; the favourite's paid 2.59 against 2.96 needed.
- Minimum-price filters (SP >= 3.0 to 6.0) gave ROI -17% to -26%; value filters
  (`p_model x SP >= 1 + margin`, on the #1 pick or on every runner) gave -11% to -20%.
  Nothing filtered turned positive.
- Model vs market: on the horses that actually won, the model gave 18.1% on average and
  the market-implied probability was 21.3%. For the model's #1 picks the model said 27.8%,
  the market 26.6%, actual 27.3%. The model is well calibrated but carries no information
  beyond the market's price.

**Odds distribution of the model's #1 picks at SP** (same 7,961 held-out races; decimal
odds, stake included):

| Group | n | Min | Q1 | Median | Q3 | Max | Mean | p90 | p99 |
|---|---|---|---|---|---|---|---|---|---|
| All bets | 7,961 | 1.04 | 2.40 | 3.50 | 5.30 | 113 | 4.66 | 8.10 | 21.0 |
| Winners | 2,175 | 1.04 | 1.83 | 2.62 | 3.75 | 71 | 3.10 | 5.00 | 10.0 |
| Losers | 5,786 | 1.12 | 2.80 | 4.00 | 6.00 | 113 | 5.25 | 9.15 | 24.0 |
| Market favourite, winners | 2,688 | 1.04 | 1.83 | 2.40 | 3.20 | 7.0 | 2.59 | 4.00 | 5.5 |

- Needed for +10%: mean winning odds of 4.03. Actual 3.10, so the return is 0.848 per
  unit and the gap is 252 units per 1,000 bets. A winner set with the same shape would
  need every winner to pay about 30% more (median about 3.4, Q3 about 4.9).
- Winners are mostly short: 59% paid under 3.0 and only 8 of 2,175 paid 15.0 or more
  (3.2% of the return). The top 5% of winners average 9.6; without them the mean is 2.77.
- Extra winners needed per 1,000 bets to close the gap if each pays 5.0 / 8.0 / 12.0 /
  21.0 / 34.0: 50 / 32 / 21 / 12 / 7 (that is +5.0 / +3.1 / +2.1 / +1.2 / +0.7 points of
  hit rate). Long-shot winners are what would move it, and the model does not find them.

**Caveats and what would change the conclusion**
- SP is not an obtainable price. Exchange best-odds or early prices have a much smaller
  takeout (about 2 to 5% plus commission) and could shift the picture; that is untested.
- Live inference has no odds at all on the Free plan (racecards/standard, which carries
  bookmaker odds, needs a Standard plan), so a live value filter cannot be run yet.
- Realistic route to a profit is extra signal the market lacks, not a price filter on this
  model. Next test: join exchange prices (or Standard-plan odds) to the held-out picks and
  repeat the table above.

## 7. Analysis: course, UK vs Ireland, field size

Sample: 117 scored races (16th to 18th; #1 pick per race, the 18th re-ranked among
actual runners). Far too small to rank individual courses.

- **UK vs Ireland: no measurable difference.** UK 20/89 = 22.5% top-1, 51.7% top-3;
  Ireland 6/28 = 21.4%, 46.4%. 95% CIs overlap almost entirely (UK top-1 0.15 to 0.32,
  top-3 0.41 to 0.62; Ireland top-1 0.10 to 0.40, top-3 0.30 to 0.64). Irish fields are
  larger (11.9 vs 9.8 runners), which alone explains the small top-3 gap. By day, UK
  33.3% / 12.9% / 22.6% top-1; Ireland 28.6% / 12.5% / 23.1%.
- **Course:** 4 to 16 races per course. Beverley 3/8, Clonmel 2/7, Downpatrick 2/8,
  Dundalk (AW) 1/5, Naas 1/8, Ayr 2/16 (12.5%), Kelso 2/5, Newbury 2/8, Newton Abbot
  2/7, Pontefract 1/7, Sandown 3/7 (43%), Southwell (AW) 2/9, Wolverhampton (AW) 1/8,
  Yarmouth 2/14 (14%): all inside noise. Do not select or avoid courses from this.
  Results mostly track average field size (Kelso 5.4 and Newton Abbot 6.7 runners look
  "good"; Ayr 12.4 looks "bad").
- **What drives results:**
  - Field size: 8 or fewer runners 31.7% top-1 / 73.2% top-3 (41 races); 9-12 18.4% /
    46.9% (49); 13+ 14.8% / 22.2% (27). UK 13+ was 1 win in 16.
  - Race type: handicaps 9/66 = 13.6% top-1 (42.4% top-3, 56% of the sample);
    maiden 33.3%, novice 41.7%, nursery 1/10, "other" 4/6, black type 1/5. Jumps 31.6%
    / 68.4% (avg field 8.4), all-weather 18.2% / 40.9%, turf flat/other 21.1% / 48.7%.
  - Confidence: the model's top-pick probability is mostly a field-size proxy. Picks
    below 0.15 won 1 of 26 (UK 0/18); 0.15-0.22 won 12/44; 0.22-0.32 won 10/34; above
    0.32 won 3/13. **Post-hoc threshold, needs out-of-sample validation.**
- **Irish naming switch** ([2.3](#23-naming-conventions-that-break-joins)): impact on
  predictions looks small because `is_ire` is about 0.01-0.04% of gain, but
  course-keyed stats (`tc_*`, `jc_*`, `hc_*`) fragment across the two spellings.
  **Fix:** normalise course names (strip `(IRE)`) in `load_raw()`, derive `is_ire` from a
  course lookup, then retrain.
- **Coverage:** 37% of Irish runners have no history in `raceform.db` vs 29% of UK
  (median 3 vs 4 prior runs, mean 9.1 vs 11.9); median gap since last run 135.5 vs 126
  days. Irish RPR coverage in history is not worse than UK (82-93% vs 71-93%, Jun 2025
  to May 2026), so data quality is not the cause of any gap.
- **Withdrawals (18th only):** 20 non-runners over 13 Irish races (1.5/race) vs 32 over
  31 UK races (1.0/race).
- **Next step:** redo this on the held-out test split in `cache/features.pkl`
  (thousands of races, `course` column is kept) for statistically usable per-course,
  per-region, per-field-size and confidence-threshold numbers.

---

## 8. Testing

- `data_collection/tests/` (`data_collection/pytest.ini`), from repo root:
  `venv/bin/python -m pytest data_collection/tests -q`. **50 pass, 4 skipped.**
  - `test_racingapi_client.py` (mocked with `responses`): credentials, error mapping
    (401 plan vs auth, 422, 429, non-JSON), throttling incl. the shared-clock
    regression, region merging, pagination stop.
  - `test_fetch_daily_racecards.py`: `racecard_to_rows()` against a real captured
    fixture (`tests/fixtures/racecards_free_gb.json`, live 2026-09-16); required fields
    present, no post-race fields, field translation, wgt/dist/region-suffix regressions.
  - `test_backfill_history.py`: mapping logic (paid shape still assumed), CLI defaults,
    plan error gives a clear message and exit code 1.
  - `test_live_db.py`: upsert idempotency against a throwaway file.
  - `test_live_smoke.py`: **opt-in** (`RUN_LIVE_API_TESTS=1`, needs the real `.env`);
    Free-tier endpoints only; also asserts the Standard-plan gate is still in effect
    (update after upgrading).
- `ml/kaggle_v2/tests/test_score_predictions.py`: `venv/bin/python -m pytest
  ml/kaggle_v2/tests -q`. **8 pass** (perfect ranking, top pick loses, non-finisher,
  unfinished race, nothing matches, int/str and float/str `num`, withdrawn top pick
  re-ranked).
- Fixtures: `racecards_free_gb.json` and `results_today_free_gb.json` (2 races each),
  `racecards_standard_401.json`, `results_paid_401.json`.
- Environment: venv at repo root (`venv/`, not committed): `python3 -m venv venv &&
  venv/bin/pip install requests python-dotenv pytest responses xgboost pandas numpy
  scikit-learn`. There is no requirements file yet.

---

## 9. Legacy system (v1) and scraper audit

**Superseded by the API and by v2.** Kept for provenance. Trial run and audit dated
2026-09-13, repo at commit `1b9c409`.

### 9.1 What v1 was

An end-to-end pipeline for predicting outcomes and identifying value bets, used live
daily; the commit history is intentionally unpolished and redundant code was stripped
for a cleanup.

- Model: trained on ~95,000 races (95,271 rows in `V2_engineered_features.csv`),
  chronological split, AUC-ROC 0.737 (95% CI 0.726-0.748), average precision 0.210
  (0.194-0.226), optimal F1 threshold 0.200, precision 21.1% vs 8.8% baseline (2.4x
  random).
- Features (20): EMA form, recent win rate (last 5), career win rate, Wilson score,
  `jt_runs`/`jt_wins`, track win rate, performance on going, age bins, weight, days
  since last race, official rating; strict point-in-time (cumulative excluding current).
- Training config: XGBoost histogram, 64/16/20 chronological split, learning rate
  0.01, max depth 5, 780 estimators; data `merged_v2_horses_labeled.csv`.
- Betting: min edge 1%, min EV 1%, strong-bet flag 15%+ edge; quarter Kelly
  (`stake = 0.25 * kelly * bankroll`); odds trend adjustment (+2% edge shortening, -2%
  drifting) from two Selenium snapshots; Lucky 15 combinations.
- Layout: `data_collection/scrape_{racecards,odds,results}.py`,
  `pipeline/{inference,betting_filters}.py`, `ml/train_model.py` (README called it
  `train.py`), `models/{xgb_model.joblib,xgb_scaler.joblib,label_encoders.pkl}`,
  `analysis/EDA*.ipynb`, `dashboard.html`.
- Daily workflow (old): scrape race cards, scrape odds, run inference, betting
  filters, scrape results next day, review `dashboard.html`.
- Known limitations: weak on horses with fewer than 2 career runs; scraper selectors
  may need updating; odds scraping needs Selenium (headless Chrome).
- Disclaimer: research/learning only; betting carries financial risk.

### 9.2 Trial-run findings (2026-09-13)

- **Environment:** system Python 3.14.7 had no ML packages and no `pip`; Arch's PEP
  668 blocks system installs, so the old README's `pip install ...` fails with
  `externally-managed-environment`. A venv at `~/venvs/horse-racing-ml` was created
  (pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, xgboost 3.4.1; ~850 MB, prebuilt
  aarch64 wheels). chromedriver + chromium present, matched at 152.0.7977.75.
- **`ml/train_model.py` runs** (well under a minute, peak RSS ~330 MB) using the
  existing engineered CSV: AUC 0.734 (CI 0.724-0.745), AP 0.201 (0.186-0.215),
  baseline precision 8.8%, close to the README's 0.737 / 0.210. Issues: writes to a
  non-existent `Pt2/models/...` (and `Pt2/xgb_precision_recall.png`,
  `Pt2/xgb_feature_importance.csv`) instead of the real `models/`, so it never
  refreshes the production model; writes `test_X.csv`/`test_Y.csv` to the repo root
  (gitignored by `*.csv` but clutter); passes the removed `use_label_encoder` param.
  `process_features()` (per-horse Python loops, ~15 min by its own print) was not
  exercised.
- **`pipeline/inference.py` broken:** needs `merged_v2_horses_labeled.csv` (not in the
  repo, likely removed by the "Autodesk cleanup" commits `f6d9c1e`, `6ecf2d7`) and
  loads its model from `Pt2/models/`. `lookup_path` reuses `V2_engineered_features.csv`
  (last-trained date 2025-10-30), a reasonable design.
- **`pipeline/betting_filters.py` broken:** chained on inference's output, and its
  default `Model_Thresholds/xgb_model_t_2026-01-14.csv` is a hard-coded past date
  (training writes today's date). Betting math (Kelly, edge/EV, Lucky 15) read as
  correct.
- Root cause: remnants of an earlier layout (`Pt2/`, "OLD" scripts, `Data_Analysis/` in
  `.gitignore`); pure path wiring, not a data or modelling problem.

### 9.3 Scraper findings (2026-09-13, `data_collection/`)

**`scrape_results.py` is 100% non-functional** (not timing-related); racecards and
odds scrapers almost certainly affected too.

- **URL typo:** uses `/racecard/{date}/` (singular); returns 406. Correct is
  `/racecards/{date}/`. Confirmed: `/racecard/2026-09-13/` 406, `/racecards/2026-09-13/`
  200, `/results/2026-09-13/` 200. The date-less branch is dead code, so every real run
  printed "Found 0 completed races to audit" (swallowed by a broad `except`).
- **Selectors dead:** Racing Post moved to Next.js with `data-testid` attributes;
  `RC-meetingItem__link`, `data-race-time`, `RC-*` and `data-test-selector` are gone.
  Working replacements: `Container__RacesSection`, `Link__Race__<raceid>` (listing);
  per runner row `Container__RunnerRowDesktop` (26/26 matched on a Curragh race) with
  `Link__Horse`, `Link__Jockey`, `Link__Trainer`, `Container__RunnerNumber`,
  `Container__HorseInfo`, `Container__RunnerStats` (OR/TS/RPR),
  `Container__RunnerRowFormFigures`, `Text__DaysSinceLastRun`, `Container__SilkAndTips`.
  The odds/price testid was never located.
- **Detail pages 406 = headless detection**, not a paywall (the `x-access-decision:
  block` / `x-user-role: free` headers were a red herring). Removing only
  `--headless` flips the same URL from 406 (5.3 KB stub) to 200 (730 KB). Plain
  `requests` can never work; a real (non-headless) browser is needed. Xvfb is installed
  (`/usr/bin/Xvfb`, `xvfb-run`) and may defeat the check for unattended runs but is
  **untested**. The anonymous results listing also showed a login link, so results
  may require authentication.
- **Rate limit (user experience, not reproduced):** roughly 50 URL hits, then a ~30
  minute block; a VPN/IP switch was used to finish a run. Design around it with
  throttling below 50, explicit block detection and checkpoint/resume.
- **Semantics bug:** `scrape_racecards.py` fetches `/racecards/` (today) while
  `scrape_odds.py` fetches `/racecards/tomorrow/`, yet both label output with
  tomorrow's date, so racecards may mislabel today's races as tomorrow's.
- No scraper code was changed. `scrape_odds.py` already uses Selenium, so if it still
  works for the user, the block is about headless/no-session access.

---

## 10. Known issues and open TODOs

One deduplicated list. **Open** unless marked fixed.

Pipeline and data:
- [x] **Upgrade to a Standard Racing API plan** (done 2026-09-20; gap loaded from exported files, see 4.6). Still to do: probe the Standard endpoints live, union `live_extension.db` into the history load, decide on `rpr`/`ts` for the gap: `backfill_history.py`
  (fills 2026-05-28 to yesterday, 114+ days and growing). Re-verify
  `results_to_rows()` against a real paid response right after (nesting, `sp_dec`,
  `ovr_btn`).
- [x] Install and enable the systemd timers (daily fetch+inference 01:00, score 23:00; done 2026-09-19, see 4.2). Backfill/evening units remain uninstalled.
- [x] Wire `live_extension.db` into inference history (done in `fast_inference.py`, 2026-09-20). Training still reads only the static export; retraining on the extended history is open.
- [ ] Verify `racecards_free(when="tomorrow")` live for the evening fetch flow.
- [ ] **Drop withdrawn horses** in the pre-race refresh before inference (free
  racecards keep non-runners; they inflate field-size features and get picked).
- [ ] Fix the sire/dam/damsire region-suffix gap (`sire_wr`/`dam_wr`/`dsire_wr` features
  are empty; needs a region source or a name-lookup workaround). Lowest priority.
- [ ] Add a requirements file for the venv (none exists).
- [ ] Reduce inference memory (peak ~2.7 GB, one OOM kill seen): downcast dtypes,
  batch, or trim history.

Model and analysis:
- [ ] Fix Irish course-name handling in `load_raw()`/`features.py` (strip `(IRE)`,
  derive `is_ire` from a lookup) and retrain; the existing "UK-only" AUC is a mixed set.
- [ ] Redo the course / region / field-size / confidence analysis on the held-out test
  split in `cache/features.pkl`; validate the "<0.15 confidence" idea out of sample.
- [ ] Score more days (about 22.6% top-1 over 115 races is a small sample); consider
  pointing `score_predictions.py` at horseracing.net for past days.
- [ ] Investigate the 2003 row-count doubling and the Nov 1995 / Oct-Nov 1996 gaps
  (only matters if pre-2015 data is used).

Legacy (low priority, superseded):
- [ ] Repoint `ml/train_model.py`/`pipeline/inference.py` from `Pt2/models/` to `models/`;
  restore or replace `merged_v2_horses_labeled.csv`; make `betting_filters.py` default to
  the latest file in `Model_Thresholds/`; drop `use_label_encoder` and the
  `test_X.csv`/`test_Y.csv` root writes; document venv install instead of system `pip`.
- [ ] Scraper (only if the API is dropped): fix the URL typo, adopt the `data-testid`
  selectors, use a non-headless browser (test Xvfb), locate the odds testid, add
  rate-limit pacing and checkpointing, fix the today/tomorrow labelling.

Fixed (kept for the record):
- [x] Sign up for the API, get a key, verify field mapping (Free plan, 2026-09-16).
- [x] Install `requests`/`python-dotenv`/`pytest`/`responses`/`xgboost`/`pandas`/`scikit-learn` in `venv/`.
- [x] Build the inference entry point (2026-09-16/17).
- [x] Per-instance rate-limit clock; `ofr`/`or` mapping; `region_codes` single value;
  1 req/s limit.
- [x] `build()` dropping all-unresolved races; weight/distance format; horse-name region
  suffix.
- [x] Full-table inference (~22 min) replaced by entity-scoped loading (~3m50s).
- [x] Scorer `num` float/string join; withdrawn-horse re-ranking.
- [x] Rows stored in finishing order (`common.load()` shuffles within race).

---

## 11. Reproduce and verify

Run from the repo root. `raceform.db` has a stray header row: always filter
`date != 'date'`; without it `MAX(date)` returns the string `'date'`.

Date range and rows for each file (expect the table in [2.1](#21-data-window-decision)):

```bash
sqlite3 data_ext/1988-2004.db "SELECT MIN(date), MAX(date), COUNT(*) FROM data WHERE date != 'date';"
sqlite3 data_ext/2005-2014.db "SELECT MIN(date), MAX(date), COUNT(*) FROM data WHERE date != 'date';"
sqlite3 data_ext/raceform.db  "SELECT MIN(date), MAX(date), COUNT(*) FROM data WHERE date != 'date';"
```

Months present per file (expect 204, 120, 137, total 461):

```bash
sqlite3 data_ext/raceform.db "SELECT COUNT(DISTINCT substr(date,1,7)) FROM data WHERE date != 'date';"
```

Gaps over 10 days between racing days (run per file; expect only the 2012 gap in
`2005-2014.db`, and the Nov 1995 / Oct-Nov 1996 ones in `1988-2004.db`):

```bash
sqlite3 data_ext/2005-2014.db "WITH d AS (SELECT DISTINCT date FROM data WHERE date != 'date'), g AS (SELECT date, LAG(date) OVER (ORDER BY date) AS prev FROM d) SELECT prev, date, CAST(julianday(date)-julianday(prev) AS INT) FROM g WHERE julianday(date)-julianday(prev) > 10;"
```

All three files in one pass (months, cross-file overlap, gaps, per-year RPR/TS/OR
coverage), from a Python with pandas:

```python
import sqlite3, pandas as pd
frames = []
for n in ["1988-2004", "2005-2014", "raceform"]:
    con = sqlite3.connect(f"data_ext/{n}.db")
    f = pd.read_sql("SELECT date, race_id, rpr, ts, [or] AS orat FROM data WHERE date != 'date'", con)
    f["src"] = n; frames.append(f)
A = pd.concat(frames); A["date"] = pd.to_datetime(A["date"], errors="coerce")
print(A.groupby("src").date.agg(["min", "max", "size"]))
print("months:", A.date.dt.to_period("M").nunique())
print("race_ids in >1 file:", int(A.groupby("race_id").src.nunique().gt(1).sum()))
print(A.assign(y=A.date.dt.year, r=pd.to_numeric(A.rpr, errors="coerce").notna()).groupby("y").r.mean().round(2))
```

Irish naming switch: which spellings exist and when:

```bash
sqlite3 data_ext/raceform.db "SELECT course, MIN(date), MAX(date), COUNT(*) FROM data WHERE date != 'date' AND course IN ('Clonmel','Clonmel (IRE)','Naas','Naas (IRE)','Downpatrick','Downpatrick (IRE)','Dundalk (AW)','Dundalk (AW) (IRE)') GROUP BY course;"
```

Live workflow:

```bash
venv/bin/python data_collection/racingapi_client.py --probe
venv/bin/python data_collection/fetch_daily_racecards.py --day today
venv/bin/python ml/kaggle_v2/inference.py --date 2026-09-18
venv/bin/python ml/kaggle_v2/score_predictions.py --date 2026-09-18   # same day only
RUN_LIVE_API_TESTS=1 venv/bin/python -m pytest data_collection/tests/test_live_smoke.py
```

---

## 12. Changelog

- **2026-09-19**: betting math and held-out ROI at starting prices (section 6.4,
  `ml/kaggle_v2/holdout_roi.py`): the model's #1 pick loses 15.2% flat at SP across 7,961
  held-out races, no price or value filter turns positive.
- **2026-09-18**: recorded the data window decision and the full-history coverage
  check (section 2.1); consolidated all markdown into this file. Course and UK-vs-Ireland
  analysis (117 races) and the Irish naming-switch finding (`087f9e1`). Scored the 18th
  through the API and fixed the scorer's `num` join and withdrawn-horse handling
  (`7c4f179`, refreshed `6a4ec34`).
- **2026-09-17**: scored the 16th and 17th against horseracing.net results (`bfd1c85`);
  added `score_predictions.py` (`d2ccfa9`).
- **2026-09-16**: real API credentials and live verification (Free plan), 45-test then
  50-test suite (`7cb850a`); inference built, entity-scoped history, wgt/dist/horse-name
  fixes (`fc39cf8`).
- **2026-09-15**: found the vendor docs via Context7 and rewrote the client and mappers
  against the real schema (`39ef44c`).
- **2026-09-14**: chose The Racing API over the scraper; scaffolded client, live DB,
  daily fetch, backfill and systemd units (`1506230`).
- **2026-09-13**: trial-run audit of the v1 pipeline and live scraper debugging
  (`FIXES.md` and `TRIAL_RUN_AUDIT.md`, now section 9); v2 (`ml/kaggle_v2`) built and run
  on the Pi.
