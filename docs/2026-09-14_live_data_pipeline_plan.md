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

## Confirmed API schema (2026-09-15 update)

`api.theracingapi.com/documentation` itself is gated behind signup and
couldn't be fetched directly, but Context7's public doc index
(`context7.com/websites/api_theracingapi`) has the same content mirrored
and fetchable without an account. Used it to pull the real endpoint/field
schema and rewrote the client + mapping functions against it (previously
they were pure guesses — see git history of this file for the original
assumed shape). Not yet verified against a *live* response with our own
key — see the residual TODO below.

**Base URL:** `https://api.theracingapi.com`
**Auth:** HTTP Basic (`Authorization: Basic <base64 user:pass>`) — Standard/Pro plans.
**Rate limits:** Free 1 req/s; Standard 5 req/s (used in the client); Pro variable.

### `GET /v1/racecards/standard` (pre-race, used by `fetch_daily_racecards.py`)
Params: `day` ("today"|"tomorrow", default "today"), `region_codes[]`, `course_ids[]`, `limit` (default 500), `skip`.
Response: `{racecards: [Race], total, limit, skip, query}`.

Race fields (subset used): `race_id, race_name, course, date, off_time, distance, field_size, going, race_class, age_band, rating_band, sex_restriction, pattern, type, runners[]`.

Runner fields (subset used) — **note the name deltas vs our schema**, already applied in `racecard_to_rows()`:
`horse, number(->num), draw, age, sex, lbs(->wgt), ofr(->or), headgear(->hg), jockey, trainer, owner, sire, dam, damsire`.
Also available but not yet consumed: `form, last_run, comment, trainer_14_days, odds[] (bookmaker quotes), silk_url, wind_surgery, colour, dob, breeder`.

### `GET /v1/results` (post-race, used by `backfill_history.py`)
Params: `start_date`, `end_date` (YYYY-MM-DD; defaults to a 365-day window if omitted — we always pass both), `region[]`, `course[]`, `type[]`, `going[]`, `race_class[]`, `limit` (default 50, we use 500), `skip`.
Paginate via `skip += limit` until `skip >= total` — `results_all_pages()` in the client does this.

Runner fields (subset used) — name deltas already applied in `results_to_rows()`:
`position(->pos), sp_dec(->sp), btn, weight_lbs(->wgt), performance_rating(->rpr), speed_rating(->ts), comments(->comment), prize, ofr(->or)`.
**Unconfirmed:** whether results rows nest under a race object with a `runners[]` array (assumed, matching racecards' shape — same vendor/doc family) or come back flat per-runner; whether `ovr_btn` (our column) has any equivalent (left NULL for now).

### Other endpoints available, not used yet
Search: `/v1/horses/search`, `/v1/jockeys/search`, `/v1/trainers/search`, `/v1/sires/search`, `/v1/dams/search`, `/v1/owners/search`.
Profile/analysis: `/v1/horses/{id}/standard`, `/v1/horses/{id}/analysis/distance-times`, `/v1/trainers/{id}/analysis/{courses,jockeys,horse-ages}`, `/v1/sires/{id}/analysis/classes`, `/v1/damsires/{id}/analysis/classes`.
Free tier (no auth): `/v1/racecards/free`, `/v1/results/today/free`, `/v1/meets/free` (North America) — useful for a quick connectivity smoke test before paying for Standard.

## ⚠️ Still unverified — do this before the first real run

The schema above is documented shape, not something confirmed against our
own account/key yet. Before relying on this pipeline:

1. Sign up (Standard plan, for the 5 req/s limit the client assumes and
   for `/racecards/standard` + `/results` access), put the key in the venv, run:
   ```
   python data_collection/racingapi_client.py --probe
   ```
2. Diff the real response against the field lists above and against
   `REQUIRED_RAW_FIELDS` in `racingapi_client.py` (mirrors
   `ml/kaggle_v2/features.py::load_raw()`'s required columns).
3. Specifically confirm: (a) `ofr` (official rating) and `sire`/`dam`/`damsire`
   are actually populated pre-race on `/racecards/standard` — they're the
   fields most likely to need a separate `/horses/{id}/standard` lookup
   instead; (b) whether `/v1/results` nests runners under races or is flat;
   (c) whether `sp_dec` is truly decimal odds (our `sp` column's expected format).
4. Fix up any drift in `racecard_to_rows()` / `results_to_rows()`.

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
