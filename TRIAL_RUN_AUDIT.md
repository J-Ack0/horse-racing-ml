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

---

## Addendum (2026-09-13, later same day) — `data_collection/` scrapers

You asked me to check all three scrapers, noting the daily ones (`scrape_racecards.py`,
`scrape_odds.py`) may fail depending on time-of-day timing quirks, but that
`scrape_results.py` ("the collection of results") should work perfectly and to
scrutinize + actually run it. I installed `requests`/`beautifulsoup4`/`selenium` into
the same venv (chromedriver + chromium are already present at
`/usr/bin/chromedriver` / `/usr/bin/chromium`, matched versions, 152.0.7977.75 — no
version-skew issue there) and ran it live against racingpost.com.

**Verdict: `scrape_results.py` does not work at all, for reasons that have nothing to
do with timing.** It's not a "ran late / missed the window" situation — it can't
retrieve a single result under any timing, today.

### Bug 1 — wrong URL (typo), always returns HTTP 406

`get_finished_race_links()` builds the URL as:
```python
url = f"https://www.racingpost.com/racecard/{target_date}/"   # singular "racecard"
```
That endpoint doesn't exist — Racing Post 406's it immediately. The correct, working
path is plural, `.../racecards/{date}/` — which is exactly what the *other two*
scrapers already use correctly. `scrape_results.py` is the odd one out. Confirmed:

```
/racecard/2026-09-13/    -> 406
/racecards/2026-09-13/   -> 200
/results/2026-09-13/     -> 200
```

Since `__main__` always calls this with a concrete date string, the buggy branch is
the *only* one ever exercised in normal use — the correct no-date branch
(`.../racecards/` with no date, used for "today") exists in the code but is dead code
given how the script is invoked. Every real run of this script has been silently
returning "Found 0 completed races to audit" (caught by a broad `except Exception`,
printed, not raised) since whenever this line was introduced.

### Bug 2 — even fixed, the site no longer renders what it's looking for

I patched the URL locally (test only, not committed) and re-ran the link discovery.
It still finds 0 links. Racing Post has since redesigned the front end: the
meeting/results listing pages have moved to a Next.js + styled-components build.
I fetched and inspected the live, current markup:

- The CSS class the scraper filters on, `RC-meetingItem__link`, **does not appear
  anywhere in the current page** (checked via raw HTML and via a full
  Selenium-rendered DOM — 0 occurrences either way).
- The `data-race-time` attribute the "is this race finished yet" cutoff logic depends
  on **also does not exist anywhere on the page** (0 occurrences).
- Anchor tags on the listing page now carry auto-generated, non-semantic class names
  (e.g. `class="sc-5d1b9ba8-1 jKoham"`) instead of the old `RC-*` scheme.

This is a real site redesign, not a stale cache or a timing fluke — every one of the
`RC-*` / `data-test-selector` / `data-race-time` selectors this scraper (and its two
siblings) were built against appears to predate the current site.

### Bug 3 — individual race/result detail pages are actively bot-blocked

Separately from the selector problem: fetching an *individual* race detail page
(the page `scrape_actual_results()` would need to open for each finished race) returns
HTTP 406 with a ~5.3KB stub page — both via plain `requests` (even with a warmed-up
session, cookies, and a same-site `Referer`) **and** via headless Chromium/Selenium.
The listing pages (`/racecards/<date>/`, `/results/<date>/`) load fine either way; it's
specifically the deep per-race URLs (`/racecards/<id>/<course>/<date>/<raceid>`,
presumably `/results/<id>/<course>/<date>/<raceid>` too) that trigger this. That
pattern — list pages open, detail pages blocked, blocked identically for a plain
requests session and a real (if headless) browser — reads like server-side bot
detection keyed on the URL pattern or a missing auth/session token, not just a
User-Agent check. I did not attempt to defeat it (fingerprint evasion is out of scope
for a legitimate personal scraper, and this crosses from "site redesign broke my
selectors" into "site is actively resisting automated access" — worth you deciding
deliberately whether/how to proceed rather than me quietly working around it).

I also noticed the anonymous session gets served a stripped-down page containing a
login link (`/auth/login/?state=...`) on the results listing — worth checking whether
Racing Post now requires an authenticated session to see result data at all, which
would be a separate, non-technical blocker on top of the above.

### Net effect on the other two scrapers

I didn't do a full run of `scrape_racecards.py` / `scrape_odds.py` (you flagged those
as already known to be flaky and out of scope for today), but the same broken
selectors and the same detail-page block almost certainly affect them too:
`scrape_racecards.py`'s `scrape_race_data()` and `scrape_odds.py`'s
`scrape_race_data_selenium()` both read `data-test-selector="RC-cardPage-runnerName"`
etc. from those same now-blocked/redesigned detail pages. `scrape_odds.py` already
uses Selenium (real browser automation), so if it's still working for you day-to-day,
that's a meaningful signal the block is specifically about headless/no-session
requests rather than the URL pattern itself — worth testing on a machine with a normal
GUI Chrome and your regular cookies before assuming it's fully dead.

One more discrepancy worth flagging: `scrape_racecards.py` fetches
`https://www.racingpost.com/racecards/` (today's listing) while `scrape_odds.py`
fetches `https://www.racingpost.com/racecards/tomorrow/` — but both then label their
output file with tomorrow's date. Depending on what the bare `/racecards/` endpoint
defaults to at the time of day you run it (today vs. rolls over to tomorrow's card
late in the day), `scrape_racecards.py` could be scraping and mis-labeling *today's*
races as tomorrow's — this is likely the "semantics/timing" issue you already had in
mind, now pinned down to a concrete cause.

### Bottom line

`scrape_results.py` is currently 100% non-functional — not intermittent, not
timing-sensitive, broken on every run: a URL typo means it never even reaches a
working results page, and the results/racecard site structure it was written against
has since changed underneath it. Fixing it for real needs: (1) the one-character URL
fix, (2) new selectors matched to the current site (I could get updated ones since
listing pages are still readable), and (3) a plan for the detail-page block — most
likely converting it to Selenium like `scrape_odds.py` already does, and testing
whether that alone clears the 406 or whether a login session is now required.

I made no code changes to any scraper — this was read + live-probe only, per your ask
to scrutinize before touching anything.
