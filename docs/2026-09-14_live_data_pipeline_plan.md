# Live Data Pipeline — Plan & Progress

**Started:** 2026-09-14
**Status:** code scaffolded, NOT YET RUN — needs API key + field-mapping verification

## Why this exists

We want to maintain daily-fresh data so the model (see `ml/kaggle_v2/README.md`
for the AUC-improvement writeup: blend_all on UK+IRE hit AUC 0.7663 vs 0.7445
on UK-only) can make real-time pre-race predictions, not just backtest on a
static export. The critical constraint driving every decision below: **feature
accessibility in real time** — whatever the pipeline supplies must be
available *before* a race runs, not reconstructed after the fact.

## Decisions made this session

1. **Source: The Racing API (theracingapi.com)**, not the Racing Post
   scraper. The scraper (`data_collection/scrape_*.py`) is confirmed broken
   — see `FIXES.md` and `TRIAL_RUN_AUDIT.md` (dead selectors after RP's
   Next.js migration, bot-blocked detail pages unless non-headless, ~50
   req/30min rate limit, one script 100% broken). The Racing API gives
   stable UK+Ireland+HK coverage, updates today's racecard every 3 minutes
   and tomorrow's every 15 minutes, rate limit 5 req/sec on the default
   plan. Matches the UK+IRE scope already trained on.
   - No racing API was documented anywhere in the repo before this session
     — confirmed via a full repo/`.md` sweep. This was a fresh decision,
     not a resumption of prior recorded work.

2. **Credentials**: user will create a venv and place the API key there
   (a `.env` with `RACING_API_USERNAME` / `RACING_API_PASSWORD`, or export
   in the venv's activate script). Code reads only from environment
   variables — nothing is hardcoded or committed.

3. **Two-script split, deliberately, to protect the point-in-time rule**:
   the README documents that all entity stats must be cumulative *up to but
   excluding* the current race date (stricter than `cumsum()-self`) —
   `common.py`'s row-shuffle fix protects a related leak. So:
   - `fetch_daily_racecards.py` — pre-race fields ONLY (no pos/sp/rpr/ts/prize).
     Runs evening (tomorrow's card, stable) and again pre-race (today's card,
     catches non-runners/declaration changes).
   - `backfill_history.py` — post-race result fields, for extending training
     history only. Structurally cannot be used for live prediction input,
     since it's a separate script/function from the racecard fetcher.
   Keeping them separate makes it structurally impossible to leak a result
   into that same day's pre-race features by accident.

4. **Separate DB, not mutating the static export**: writes go to
   `data/live_extension.db` (new, same `data` table schema as
   `data/raceform.db`), not into the Kaggle export itself. Verified via
   `sqlite3`: `data/raceform.db` covers `2015-01-01` → `2026-05-27`
   (1,851,286 rows). `live_extension.db` starts at `2026-05-28`.

## What got written this session

- `data_collection/racingapi_client.py` — HTTP Basic auth client, rate
  throttled to 5 req/sec, `.racecards(day)` and `.results(day)` methods,
  `--probe` mode to dump a live response for field verification.
- `data_collection/live_db.py` — shared SQLite writer, upserts on
  `(date, race_id, horse)`, same columns as `raceform.db` plus a
  `fetched_at` audit column.
- `data_collection/fetch_daily_racecards.py` — daily pre-race fetch job.
- `data_collection/backfill_history.py` — pulls settled results day-by-day
  from `2026-05-28` (dataset end + 1) through yesterday; idempotent, safe
  to re-run; defaults recompute the gap automatically so a daily re-run
  self-heals any day that failed.
- `data_collection/systemd/*.service` + `*.timer` — evening racecard fetch
  (18:00) and morning backfill (07:00), following this Pi's existing
  pattern of systemd user timers rather than cron (no crontab is set up
  here — see envmap). NOT yet installed/enabled.

## ⚠️ Unverified — must confirm before first real run

The Racing API's documentation page (`api.theracingapi.com/documentation`)
is gated behind signup; I could not fetch the actual field schema. Every
script above encodes **assumed** JSON shapes and field names
(`racecards_to_rows()` / `results_to_rows()` docstrings flag this
explicitly). Before relying on this pipeline:

1. Sign up, put the key in the venv, run:
   ```
   python data_collection/racingapi_client.py --probe
   ```
2. Compare the real response against `REQUIRED_RAW_FIELDS` in
   `racingapi_client.py` (mirrors `ml/kaggle_v2/features.py::load_raw()`'s
   required columns: date, course, race_id, off, race_name, type, class,
   pattern, age_band, sex_rest, dist, going, ran, num, pos, draw, horse,
   age, sex, wgt, hg, jockey, trainer, or, sire, dam, damsire).
3. Fix up the key names in `racecard_to_rows()` (fetch_daily_racecards.py)
   and `results_to_rows()` (backfill_history.py) to match reality.
4. Confirm official rating (`or`) and breeding fields (`sire`/`dam`/`damsire`)
   are actually present pre-race on the racecard endpoint — these are the
   fields most likely to require a separate horse-profile lookup rather
   than being inline on the racecard.

## Not done yet (next steps)

- [ ] Sign up for The Racing API, get key, verify field mapping (above).
- [ ] Run `backfill_history.py` once to fill `2026-05-28` → yesterday.
- [ ] Install/enable the systemd timers (`systemctl --user enable --now ...`).
- [ ] Wire `live_extension.db` into `ml/kaggle_v2/common.py` / `features.py`
      so the feature build reads a UNION of `raceform.db` + `live_extension.db`
      (not done — those files are currently untouched, still point only at
      the static export).
- [ ] Add `requests` (and `python-dotenv` if wanted) to whatever
      requirements file the venv installs from — not present in the base
      environment (`ModuleNotFoundError: requests`, checked this session).
- [ ] Decide inference entry point: a script that, given today's
      `live_extension.db` rows, builds features and calls the trained
      model — not built yet, this session only covers data acquisition.

## Cross-references

- `ml/kaggle_v2/README.md` — AUC improvement writeup, point-in-time rules.
- `FIXES.md`, `TRIAL_RUN_AUDIT.md` — why the RP scraper was rejected.
- `data/Kaggle_ReadMe.md` — static dataset provenance.
