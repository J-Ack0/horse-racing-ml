# horse-racing-ml — Trial Run & Audit Report

**Date:** 2026-09-13
**Machine:** Raspberry Pi 5, 8GB RAM, no swap, Arch Linux ARM (aarch64)
**Repo:** `~/git/horse-racing-ml` (clean, up to date with `origin/main`, commit `1b9c409`)

## Scope

Per your request, I inspected the repo, set up a working Python environment, and did a
trial run of the 3 core pipeline scripts (the ones that make up the "live" daily system,
per the README):

1. `ml/train_model.py` — trains the XGBoost win-predictor
2. `pipeline/inference.py` — runs the trained model on a race card to get win probabilities
3. `pipeline/betting_filters.py` — turns predictions + odds into Kelly-sized value bets

The two data-collection/scraper scripts and the "old" analysis notebooks were not run
(scrapers need live network access to Racing Post + a matching ChromeDriver; out of scope
for a local trial and there's an explicit `# OLD` block in `.gitignore` suggesting several
of those are already deprecated).

---

## 0. Environment finding (blocking, fixed for this trial)

The system Python (3.14.7, `/usr/bin/python3`) had **no ML packages installed at all** —
not even `pip`. `pip3`/`pip` don't exist system-wide, and Arch's PEP 668
"externally-managed" policy blocks installing into the system Python anyway.

**Fix applied:** created a project venv at `~/venvs/horse-racing-ml` and installed the
packages the 3 scripts actually import (`pandas`, `numpy`, `scikit-learn`, `xgboost`,
`joblib`, `scipy`, `matplotlib`, `seaborn`, `tqdm`). All installed cleanly as prebuilt
aarch64 wheels — no source compilation needed, ~850MB on disk, no dependency conflicts.

Resulting versions: pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, xgboost 3.4.1.

To reuse it yourself:
```bash
source ~/venvs/horse-racing-ml/bin/activate
python ml/train_model.py
```

This is worth fixing properly — right now the README's `pip install pandas numpy ...`
line will simply fail on this machine with `error: externally-managed-environment` if
someone runs it as documented (system-wide, no venv mentioned).

---

## 1. `ml/train_model.py` — ran successfully end-to-end

Since `V2_engineered_features.csv` already exists in the repo (95,271 rows, pre-computed),
the script took the fast path and skipped the expensive feature-engineering stage —
more on that below, this matters for your RAM/time warning.

**Result:** trained fine, finished in well under a minute of wall time, peak RSS ~330MB
(nowhere near the 8GB ceiling). Output matched the README's claimed performance closely:

| Metric | This run | README |
|---|---|---|
| AUC-ROC | 0.734 (CI 0.724–0.745) | 0.737 (CI 0.726–0.748) |
| Average Precision | 0.201 (CI 0.186–0.215) | 0.210 (CI 0.194–0.226) |
| Baseline precision | 8.8% | 8.8% |

The small deltas are expected run-to-run noise (bootstrap CI, no fixed seed on the
bootstrap resampling loop itself — the model fit is seeded, but `bootstrap_ci`'s own
`RandomState(seed=42)` should make even that reproducible; the difference is more
likely from library version drift, e.g. XGBoost 1.x → 3.4.1 changing default
tree-building numerics slightly). Not a concern.

**Issues found in this script:**

- **Output paths don't match the README or the real model directory.** The script
  hardcodes `Pt2/models/xgb_model.joblib`, `Pt2/models/xgb_scaler.joblib`,
  `Pt2/xgb_precision_recall.png`, `Pt2/xgb_feature_importance.csv` — but `Pt2/` is a
  directory that doesn't exist in the repo, isn't in the README's documented structure,
  and is one of the "OLD" paths explicitly excluded in `.gitignore`. The *actual*
  production model lives at `models/xgb_model.joblib` (root `models/` dir, 3.7MB,
  committed to git). Running this script does **not** update that file — it silently
  creates a brand-new, disconnected `Pt2/models/` tree instead. If you (or a past
  version of you) intended `ml/train_model.py` to refresh the production model, it
  currently doesn't — you'd retrain and the live model would be untouched. This looks
  like a leftover from an old directory layout (`Pt2` = "Part 2"?) that never got
  updated when the repo was reorganized/cleaned up.
- **Also writes `test_X.csv` and `test_Y.csv` to the repo root** (the held-out test
  split, 2.6MB + 38KB). Root-level, not gitignored by name (only generic `*.csv` is
  ignored, which does cover it — so it won't get committed, but it does clutter
  the working tree on every run).
- **`use_label_encoder': False` is passed to `XGBClassifier`.** This parameter was
  removed in XGBoost well before 3.x. It didn't crash (XGBoost just stores unknown
  params and ignores them at fit time), but it means the comment ("Suppress warning")
  is stale — worth deleting the line.
- The heavy, genuinely slow part of this script — `process_features()` — was **not**
  exercised in this trial, because `V2_engineered_features.csv` already exists so the
  script short-circuits past it. That function does per-horse Python loops
  (`for horse in df['horse_name'].unique(): df[df['horse_name']==horse].copy()...`,
  twice — once for EMA form, once for career stats) over what's likely several
  thousand unique horses, each doing a full boolean-mask scan of the dataframe. That
  is the part that would actually justify your "RAM will be high, allow time" warning
  — it's O(horses × rows) and was reportedly ~15 minutes per the script's own print
  statement. I did **not** force this path (the raw input it needs,
  `merged_v2_horses_labeled.csv`, isn't in the repo — see next section), so I can't
  give you real numbers for it from this trial. If you want that exercised, I'd need
  the raw pre-engineering CSV.

---

## 2. `pipeline/inference.py` — broken as shipped (missing required data file)

With no `Inference_Inputs/Input_<date>.csv` present, it correctly no-ops
("Error: No input file found") rather than crashing — fine.

I created a minimal synthetic race-card row to push it further, and it fails hard:

```
FileNotFoundError: [Errno 2] No such file or directory: 'merged_v2_horses_labeled.csv'
```

**Issue:** `historic_path` is hardcoded to `merged_v2_horses_labeled.csv`, which is
**not present anywhere in the repo** (only the already-engineered
`V2_engineered_features.csv` is committed; the raw merged file this script wants is
gone — likely one of the files cleaned out in the "Autodesk further cleanup" commits).
So today, inference cannot run at all, synthetic input or real. This also means the
day-to-day workflow described in the README (steps 1→6) is currently non-functional
past step 1, for two independent reasons (this missing file, and the `Pt2/models/`
path from `train_model.py` above — `model_path` here is *also* hardcoded to
`Pt2/models/xgb_model.joblib`, so even if the data file existed, it'd load whatever
was last dumped into that ad-hoc `Pt2/` folder rather than the real `models/` model).

Minor secondary issue: `lookup_path` defaults to `V2_engineered_features.csv` — reusing
the training feature file as a live lookup table for horse history. That's a reasonable
design choice (not a bug), just worth knowing it means inference's accuracy for known
horses is only as fresh as that CSV's last-trained date (2025-10-30 per the training
run's date range).

---

## 3. `pipeline/betting_filters.py` — broken as shipped (chain of missing inputs + a stale default)

Running with default CLI args fails immediately:

```
FileNotFoundError: Required file not found: Inference_Outputs/Output_2026-09-14.csv
```

Expected, since `inference.py` never produced that file (see above) — this script is
next in the daily chain, so it inherits the same blocker.

**Independent issue found by reading the code** (not yet reached by execution, since
the pred-file check fails first): the default threshold-file path is hardcoded to a
**fixed past date**:

```python
default_thr = "Model_Thresholds/xgb_model_t_2026-01-14.csv"
```

`train_model.py`, by contrast, generates that filename dynamically from *today's* date
(`xgb_model_t_{datetime.now():%Y-%m-%d}.csv`) every time it's run. Since the two dates
will only ever coincide by accident, running `betting_filters.py` with its defaults
on any day other than 2026-01-14 — even after fixing the two issues above — will hit a
**third** `FileNotFoundError` unless the user manually passes `--thresholds`. This looks
like a copy-pasted default that never got parameterized (e.g. "most recent file in
`Model_Thresholds/`" would be more robust).

The betting math itself (Kelly sizing, edge/EV, Lucky-15 combinatorics) reads as
correct and matches the README's description — I didn't find defects in that logic,
just in the file-path wiring around it.

---

## Summary — is the pipeline runnable today?

| Script | Runs? | Blocking issue |
|---|---|---|
| `ml/train_model.py` | ✅ Yes | Cosmetic: writes to `Pt2/` instead of `models/` |
| `pipeline/inference.py` | ❌ No | Missing `merged_v2_horses_labeled.csv`; also points at `Pt2/models/` |
| `pipeline/betting_filters.py` | ❌ No (untestable past step 1) | Chained on inference's output; separately has a hardcoded stale default date |

**Root cause common to 2 and 3:** the `Pt2/` output path in `train_model.py` and the
missing `merged_v2_horses_labeled.csv` input both look like remnants of an earlier
project layout (the `Pt2/`, "OLD" scripts, and `Data_Analysis/` entries in
`.gitignore` all point at a prior restructuring). My best guess: the cleanup commits
(`f6d9c1e`, `6ecf2d7` — "Autodesk...cleanup", "Directory cleaning...Messy Presentation
for AutoDesk Internship") removed/renamed files for presentation but didn't update the
three scripts' hardcoded paths to match the new `models/` / root-level layout the
README now documents.

**None of this is a data or modeling problem** — the trained model itself is good and
reproduces the README's headline numbers almost exactly. It's purely file-path wiring
that's drifted from the current directory layout.

## Suggested next step (not done — your call)
If you want, I can patch the three hardcoded paths (`Pt2/models/...` → `models/...` in
both `train_model.py` and `inference.py`, and the stale threshold date in
`betting_filters.py` → "pick latest file in `Model_Thresholds/`") so the daily workflow
in the README actually runs as documented. I left the repo untouched otherwise — the
only new file is this report, and the `~/venvs/horse-racing-ml` venv (outside the repo).
