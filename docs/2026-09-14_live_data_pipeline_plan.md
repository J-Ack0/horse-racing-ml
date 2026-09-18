# Live Data Pipeline — Plan & Progress

**Started:** 2026-09-14
**Last updated:** 2026-09-16
**Status:** LIVE-VERIFIED. Live prediction data pull works end-to-end today
(no plan upgrade needed). Historical backfill is code-complete but blocked
on a Standard plan subscription. 45-test suite in place, all passing
(41 mocked + 4 live).

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

## Live-verified 2026-09-16

Credentials were added to a repo-root `.env` (`USERNAME=`/`PASSWORD=`,
gitignored). Built a venv (`venv/`, not committed) with
`requests`/`python-dotenv`/`pytest`/`responses`, then ran the pipeline
against the real API — not just against vendor docs. Findings that changed
the design:

1. **Account plan is Free, not Standard.** `/racecards/standard` and the
   historical `/v1/results` both return `401 {"detail": "Standard Plan
   required"}`. This was unknown until tested live.
2. **The Free racecards endpoint (`/racecards/free`) is actually sufficient
   for live prediction.** Live-pulled response has every pre-race field
   `features.py` needs: `ofr` (official rating), `lbs` (weight), `draw`,
   `sire`/`dam`/`damsire`, `headgear`, `jockey`, `trainer`, `owner`, plus
   race-level `race_class`, `going`, `age_band`, `sex_restriction`,
   `pattern`, `rating_band`. **No plan upgrade needed to keep the daily
   racecard fetch running.**
3. **Historical backfill (`backfill_history.py`) is blocked**, not just
   unverified — confirmed 401 live. `/results/today/free` (the free
   results tier) does NOT substitute: it lacks `sp`/`rpr`/`ts`/`prize`/
   `comment` entirely, even for just today. A Standard plan subscription is
   a hard requirement to fill the `2026-05-28` → yesterday gap.
4. **`region_codes` takes exactly one region per call**, not a combined
   value — `"gb+ire"` 422s ("unrecognised region code"). Both
   `racecards_free()` and `racecards_standard()` now loop `REGIONS = ["gb",
   "ire"]` and merge.
5. **Free-plan rate limit is 1 req/sec, not 5** — confirmed via a live 429
   ("Rate limit exceeded: 1 per 1 second"). `RATE_LIMIT_PER_SEC` corrected.
6. **Endpoint field-name inconsistency, caught by the test suite, not
   manual inspection**: `/racecards/free` uses `ofr` for official rating,
   but `/results/today/free` uses `or` for the same value on the same
   account. `backfill_history.py::results_to_rows()` now tries both
   (`runner.get("or", runner.get("ofr"))`).
7. **Unrated horses return the literal string `"–"` (en dash)** for
   `or`/`ofr`, not null or 0 — anything doing numeric parsing on that
   column downstream must handle it.
8. **Rate-limit clock was a real bug, not just a test artifact**: it was
   stored per-`RacingAPIClient`-instance (`self._last_request_ts`), so two
   client objects created back-to-back (e.g. two separate script
   invocations sharing a process, or the daily fetch + backfill running in
   sequence) would each think they got a fresh 1-req/sec allowance and
   double the real rate against the account. Fixed to a class-level shared
   clock (`RacingAPIClient._last_request_ts`), with a regression test
   (`test_rate_limit_clock_is_shared_across_instances`).

**Live pull confirmed working end-to-end**: `fetch_daily_racecards.py --day
today` wrote 327 real runner rows across 5+ GB/IRE courses into
`data/live_extension.db` with every required field populated.

### Test suite

`data_collection/tests/` — 45 tests, `pytest.ini` at `data_collection/`
root. Run with `venv/bin/python -m pytest` from `data_collection/`.

- `test_racingapi_client.py` (mocked via `responses`): credential handling,
  error-code mapping (401/plan-vs-auth/422/429/non-JSON bodies), rate-limit
  throttling incl. the shared-clock regression test, region merging,
  pagination stop condition.
- `test_fetch_daily_racecards.py`: `racecard_to_rows()` against a REAL
  captured fixture (`tests/fixtures/racecards_free_gb.json`, live
  2026-09-16) — checks every `REQUIRED_RAW_FIELDS` entry is present, no
  post-race field leaks in, field-name translation is correct.
- `test_backfill_history.py`: `results_to_rows()` mapping logic (still
  bounded by point 3 above — full field coverage needs a paid-plan
  fixture), CLI date-range defaults, and that a plan error surfaces a clear
  message + exit code 1 instead of crashing.
- `test_live_db.py`: upsert-on-`(date, race_id, horse)` idempotency,
  declaration-change overwrites, missing-key-becomes-NULL, schema
  re-entrancy — all against a throwaway sqlite file, never the real
  `live_extension.db`.
- `test_live_smoke.py`: **opt-in**, skipped unless
  `RUN_LIVE_API_TESTS=1` (needs the real `.env`) — hits only Free-tier
  endpoints, confirms the live schema hasn't drifted from the fixtures, and
  confirms the Standard-plan gate is still in effect (update/remove that
  assertion after upgrading).

Three real bugs were caught and fixed by writing these tests, not by
inspection: the `ofr`/`or` field-name inconsistency (#6 above), the
per-instance rate-limit clock (#8 above), and two of the tests' own
assumptions were initially too strict (checked `__dict__` contents instead
of repr; required non-empty `pattern`/`sex_rest` when empty string is a
legitimate value for a non-pattern race) — fixed in the tests themselves.

## Not done yet (next steps)

- [x] ~~Sign up for The Racing API, get key, verify field mapping.~~ Done —
      key is live, Free plan, mapping verified against real responses.
- [ ] **Upgrade to Standard plan** to unblock `backfill_history.py` — this
      is now the single blocker on filling `2026-05-28` → yesterday.
      Re-verify `results_to_rows()`'s field mapping against a real paid
      response immediately after upgrading (see point 3, live-verified
      section) — it is currently the assumed/documented shape, not
      confirmed.
- [ ] Install/enable the systemd timers (`systemctl --user enable --now ...`)
      — units exist in `data_collection/systemd/`, not yet installed.
- [ ] Wire `live_extension.db` into `ml/kaggle_v2/common.py` / `features.py`
      so the feature build reads a UNION of `raceform.db` + `live_extension.db`
      (not done — those files are currently untouched, still point only at
      the static export).
- [x] ~~Add `requests`/`python-dotenv` somewhere installable.~~ Done —
      `venv/` created at the repo root with `requests`, `python-dotenv`,
      `pytest`, `responses` (not committed; recreate with `python3 -m venv
      venv && venv/bin/pip install requests python-dotenv pytest responses`).
- [ ] Decide inference entry point: a script that, given today's
      `live_extension.db` rows, builds features and calls the trained
      model — not built yet, this session only covers data acquisition
      and its test coverage.
- [ ] `racecards_free(when="tomorrow")` is untested live — the evening
      fetch flow (pulling tomorrow's card) hasn't been confirmed to
      actually return data for "tomorrow" on the Free plan; only "today"
      has been live-verified.

## Inference built and run live: 2026-09-16/17

Built `ml/kaggle_v2/inference.py` — scores a day's racecard with the base
(UK+IRE combined, includes `is_ire`) blend_all model: the three saved
boosters `cache/final_{binary,softmax,pltop3}.json`, blended by the exact
geometric-mean-then-renormalise method `exp_final.py`'s `blend()` uses.
Output: `ml/kaggle_v2/predictions/predictions_<date>.csv` (gitignored, it's
a run artifact) plus a top-pick-per-race console summary.

**Performance**: the first working run (full ~1.85M-row historical table,
`HISTORY_START=2021-01-01` → ~900K rows) took ~22 minutes on this Pi. Fixed
per user direction — inference only needs each of today's ~150-450
horses'/jockeys'/trainers'/sires'/dams'/damsires' OWN history, not a full
scan, since every rolling/entity stat in `features.py` (`day_stats`, a
horse's own `prior_*` shifts) is computed per-entity. `inference.py` now
runs 6 separate indexed queries (`data_ext/raceform.db` got new indexes on
`date` + all six entity columns) — one per entity column, `entity IN
(today's values)` — unions and dedupes them, instead of a date-range scan.
Result: **~1.85M rows → ~780K rows loaded → ~3m50s** end to end (still
dominated by `features.build()`'s groupby/cumsum work over that row count,
not the SQL read). `features.py::build()` also got a defragmenting
`df.copy()` after the `day_stats` loop (pure performance, verified
behavior-preserving).

**Real correctness bugs found only by running this live, not by
inspection** (all fixed, all covered by new regression tests in
`data_collection/tests/test_fetch_daily_racecards.py`):

1. `features.py::build()` dropped every race with zero winners so far —
   correct for a data-integrity check on settled historical races, but it
   silently deleted ALL of today's rows (none have run yet). Fixed to keep
   a race when either it has ≥1 winner OR every row in it is unresolved
   (`any_result_per_race == 0`).
2. The API gives weight in plain lbs (`"140"`) and distance in plain
   furlongs (`"10.0"`), but `features.py`'s `parse_wgt()`/`parse_dist()`
   expect raceform.db's string formats (`"10-0"` stone-lbs, `"1m2f"`) and
   silently return NaN otherwise — not an error, just empty features. Fixed
   with `lbs_to_wgt_str()`/`furlongs_to_dist_str()` in
   `fetch_daily_racecards.py`, applied at row-construction time (not by
   touching the shared parser, to keep the training path untouched).
3. **The big one**: `data_ext/raceform.db` suffixes EVERY horse name with
   its region in parens — `"Great Blasket (IRE)"`, and even GB-bred horses
   get `"(GB)"`, zero exceptions found. The API's `horse` field is bare. Since
   `features.py::build()` keys its per-horse `groupby('horse')` on that exact
   string, a mismatch means today's row and that same horse's own
   historical rows are treated as two unrelated entities — EVERY
   `prior_*`/`h_*` feature (win rate, days since last run, prior RPR/TS/OR,
   course/distance/going experience) comes back NaN for every live runner,
   regardless of how much history is loaded, because the join key itself
   never matches. This produced 30 all-NaN features on the first correct
   run and wasn't caught by any test — it doesn't raise, it just produces
   silently-empty features. Fixed with `with_region_suffix()`, using the
   API's per-runner `region` field (confirmed present on the Free tier).
   **Known residual gap**: `sire`/`dam`/`damsire` are horse names too and
   have the exact same suffix convention, but the Free-tier racecard
   response does not include `sire_region`/`dam_region`/`damsire_region`
   (only `region` for the horse itself is present) — so `sire_wr`/`dam_wr`/
   `dsire_wr`-derived features are still silently zero/NaN for now. Lowest
   priority to fix of the three (breeding-based stats, much less predictive
   than the horse's own form) but not yet done.

Final clean run (2026-09-17, 444 runners / 39 races): only 1 feature
(`sire_wr_z`, an expected consequence of the residual sire/dam/damsire gap
above) came back all-NaN, down from 30.

## Live scoring vs actual results (2026-09-16 and 2026-09-17)

**Free historical-results source found**: `https://www.horseracing.net/results/<course>/<dd-mm-yy>`
(e.g. `/results/yarmouth/16-09-26`) is server-rendered and returns a full
card with 1st/2nd/3rd per race via WebFetch. Racing Post, Sporting Life,
AtTheRaces, RacingTV and BBC results pages do not (JS apps / blocked, same
finding as `FIXES.md`). This gets around the Standard-plan gate for
*scoring* only (top 3, no full finishing order, no sp/rpr/ts), so it
cannot substitute for `backfill_history.py`. It also means the "score the
same day via `/results/today/free`" deadline in `score_predictions.py` is
not hard: yesterday's results are still recoverable this way.

| Day | Races | Top-1 precision (#1 pick won) | #1 pick finished top 3 | Precision@3 (model's top 3 vs actual top 3) |
|---|---|---|---|---|
| 2026-09-16 | 34 | 11/34 = 32.4% | 22/34 = 64.7% | 50/102 = 49.0% |
| 2026-09-17 | 39 | 5/39 = 12.8% | 15/39 = 38.5% | 37/117 = 31.6% |
| Combined | 73 | 16/73 = 21.9% | 37/73 = 50.7% | 87/219 = 39.7% |

- Training-time reference (`results_final.csv`): top-1 ~0.27, top-3 (winner in
  model's top 3) ~0.60. Random baseline at ~11 runners/race is ~9% top-1.
- The 16th ran on the degraded feature set (before the horse-name region
  fix, ~25 features all-NaN); the 17th ran on the fixed set. The 17th was
  *worse*, so the fix did not visibly help, but n=39 is far too small to
  separate that from day-to-day variance (top-1 SE at n=39 is ~7 points).
- An earlier note said 35 races for the 16th; the correct count is 34 (one
  Kelso race was dropped by `build()`, 3 of 327 runners).
- Mean position error is not computable from top-3-only results (needs full
  finishing orders). Both days still carry the ~113-day history gap and the
  sire/dam/damsire suffix gap.

## 2026-09-18 scored via the API (score_predictions.py)

Inference was run at ~19:30 IST, after most races had finished, so this is a
post-hoc run (the model never sees results, so no leakage, but it is not a
prospective pre-race test). `/results/today/free` returned 42 finished races
(416 runners, full finishing positions incl. `PU`/`F`).

| Races | Top-1 (#1 pick won) | #1 pick finished top 3 | Precision@3 | Mean position error |
|---|---|---|---|---|
| 44 | 10/44 = 22.7% | 22/44 = 50.0% | 60/132 = 45.5% | 3.36 (random-order ≈ 3.5) |

(Re-run later the same evening once two more races had finished; the first
pass over 42 races gave 23.8% / 52.4% / 45.2% / 3.34.)

- **Results API carries positions but no features**: runner fields are only
  `position`, `number`, names/ids, `weight`/`weight_lbs`, `or`, `draw`,
  `headgear`, `sex`, `age`, breeding. No sp, rpr, ts, prize, comment, btn or
  time. So it can score predictions but cannot feed `prior_rpr`/`prior_ts`
  history; still no backfill without a Standard plan.
- **Withdrawn horses (new pipeline finding)**: `/racecards/free` keeps
  non-runners. 6 of 42 races had a withdrawn #1 pick and 51 declared runners
  never ran. `score_predictions.py` now re-ranks each race among actual runners
  (`n_nonrunners_dropped` in its output). For prospective use, the pre-race
  refresh must drop withdrawn horses before inference, otherwise the model
  ranks horses that will not run and field-size features are inflated.
- Scorer bug fixed: predictions store `num` as float (`2.0`), API as string
  (`"2"`), so the first run matched 0 races; both are now normalised (test added).
- Mean position error is only slightly better than random ordering, so the
  model's value is in the top of the ranking, not the full order.
- Cumulative (16th + 17th + 18th, different scoring paths): 26 wins in 115
  races = 22.6% top-1; the 18th's top-3 is 52.4%. History gap now 114 days.

## Qualitative analysis: course selection and UK vs Ireland (2026-09-18)

Sample: 117 scored races (16th, 17th, 18th; #1 pick per race, 18th re-ranked
among actual runners). Far too small to rank individual courses.

- **UK vs Ireland: no measurable difference.** UK 20/89 = 22.5% top-1, 51.7%
  top-3; Ireland 6/28 = 21.4%, 46.4%. 95% CIs overlap almost entirely
  (Ireland top-1 CI 0.10 to 0.40). Irish fields are larger (11.9 vs 9.8
  runners), which alone explains the small top-3 gap.
- **Course**: 4 to 16 races per course; Sandown 43%, Ayr 12.5%, Yarmouth 14%,
  Wolverhampton 12.5% are all inside noise. Do not select or avoid courses
  from this. Course results mostly track average field size (Kelso 5.4 and
  Newton Abbot 6.7 runners look "good"; Ayr 12.4 looks "bad").
- **What does drive results**: field size (<=8 runners: 31.7% top-1 / 73.2%
  top-3; 13+: 14.8% / 22.2%; UK 13+ was 1/16) and race type (handicaps 9/66 =
  13.6% top-1 and 56% of the sample; maiden/novice 33 to 42%; nurseries 1/10).
  The model's own top-pick probability is mostly a field-size proxy; picks
  below 0.15 won 1 of 26 (post-hoc threshold, needs out-of-sample validation).
- **Structural Irish data issue (found this session)**: from 2025-10-15 the
  source data dropped the "(IRE)" suffix from Irish course names ("Naas (IRE)"
  became "Naas"). `features.py` sets `is_ire` from that suffix, so Irish
  runners after that date, and every live Irish race, get `is_ire = 0`. The
  README's "Irish data ends 2025-10-14" is a labelling artifact: Irish racing
  continues in `raceform.db` under bare names (e.g. Dundalk 3,166 rows since).
  The UK+IRE model's "UK-only" test window (2025-11-29 to 2026-05-27)
  contained 13,818 Irish runners in 1,200 races, so the 0.7445 blend_all AUC
  was on a mixed set, not UK only. Impact on predictions looks small:
  `is_ire` is ~0.01 to 0.04% of model gain (rank ~120 of 131); course-keyed
  stats (`tc_wr` ~0.9%, `jc_wr` ~0.5%) fragment across the two spellings.
  The models lean on the horse's own form: `prior_rpr` rank/z-score, `h_rel_ema`,
  field size. Fix: normalise course names (strip "(IRE)") in `load_raw()` and
  derive `is_ire` from a course lookup, then retrain.
- **Coverage**: 37% of Irish runners have no history in `raceform.db` vs 29% of
  UK (median 3 vs 4 prior runs). History for both stops 2026-05-27
  (median ~4 months since last run). Irish RPR coverage in history is not worse
  than UK (82 to 93% vs 71 to 93%), so data quality is not the cause of any gap.
- **Withdrawals (18th only)**: 20 non-runners over 13 Irish races (1.5/race)
  vs 32 over 31 UK races (1.0/race).
- **Suggested next step**: redo this on the held-out test split in
  `cache/features.pkl` (thousands of races, `course` column is kept) to get
  statistically usable per-course, per-region, per-field-size and
  confidence-threshold numbers.

## Cross-references

- `ml/kaggle_v2/README.md` — AUC improvement writeup, point-in-time rules.
- `FIXES.md`, `TRIAL_RUN_AUDIT.md` — why the RP scraper was rejected.
- `data/Kaggle_ReadMe.md` — static dataset provenance.
- `ml/kaggle_v2/inference.py` — live scoring entry point.
