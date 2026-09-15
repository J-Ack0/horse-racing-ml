"""
One-off (then daily) backfill: pull SETTLED RESULTS day-by-day and write them
into live_extension.db, so the gap between the static Kaggle export's last
date and "yesterday" gets filled in for rolling/entity feature history.

The static export (data/raceform.db) currently ends 2026-05-27 (verified via
`SELECT MIN(date), MAX(date) FROM data` — see
docs/2026-09-14_live_data_pipeline_plan.md). Default start is the day after
that; override with --start if the export gets refreshed later.

This writes POST-race fields (pos, sp, rpr, ts, prize) which is fine here —
this script backfills TRAINING history, it is never used for the live
pre-race feature build (fetch_daily_racecards.py does that, pre-race fields
only). Keeping the two scripts separate is deliberate: it makes it
structurally impossible to leak a result into a same-day pre-race feature.

The API's /v1/results endpoint takes a start_date/end_date range directly
(not one call per day) and paginates at limit=500 rows/page — see
docs/2026-09-14_live_data_pipeline_plan.md, "Confirmed API schema".

Usage:
    python backfill_history.py                      # 2026-05-28 -> yesterday
    python backfill_history.py --start 2026-06-01 --end 2026-06-30
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta, datetime, timezone

from racingapi_client import RacingAPIClient, RacingAPIError
from live_db import connect, upsert_rows

DATASET_END_DATE = date(2026, 5, 27)  # last date present in data/raceform.db


def results_to_rows(payload: dict, fetched_at: str) -> list[dict]:
    """
    Map one /v1/results page onto our `data` table row shape.

    Field names per the documented schema (see docs/2026-09-14_live_data_pipeline_plan.md):
    API `position` -> our `pos`, `sp_dec` -> our `sp` (decimal, matches how
    raceform.db's `sp` column is used downstream), `weight_lbs` -> our `wgt`,
    `performance_rating`/`speed_rating` -> our `rpr`/`ts`, `comments` -> our
    `comment`, `race_class` -> our `class`. `ovr_btn` has no confirmed API
    equivalent (only overall `btn` documented) — left NULL until verified.
    Response is assumed race-grouped with nested runners, mirroring
    /racecards/standard's shape (same vendor, same doc family) — CONFIRM
    with a --probe once results() is first called for real.
    """
    rows: list[dict] = []
    for race in payload.get("results", []):
        base = {
            "date": race.get("date"),
            "course": race.get("course"),
            "race_id": race.get("race_id"),
            "off": race.get("off_time"),
            "race_name": race.get("race_name"),
            "type": race.get("type"),
            "class": race.get("race_class"),
            "pattern": race.get("pattern"),
            "rating_band": race.get("rating_band"),
            "age_band": race.get("age_band"),
            "sex_rest": race.get("sex_restriction"),
            "dist": race.get("distance"),
            "going": race.get("going"),
            "ran": race.get("field_size") or len(race.get("runners", [])),
            "fetched_at": fetched_at,
        }
        for runner in race.get("runners", []):
            row = dict(base)
            row.update({
                "num": runner.get("number"),
                "draw": runner.get("draw"),
                "pos": runner.get("position"),
                "btn": runner.get("btn"),
                "horse": runner.get("horse"),
                "age": runner.get("age"),
                "sex": runner.get("sex"),
                "wgt": runner.get("weight_lbs"),
                "hg": runner.get("headgear"),
                "sp": runner.get("sp_dec"),
                "jockey": runner.get("jockey"),
                "trainer": runner.get("trainer"),
                "prize": runner.get("prize"),
                "or": runner.get("ofr"),
                "rpr": runner.get("performance_rating"),
                "ts": runner.get("speed_rating"),
                "sire": runner.get("sire"),
                "dam": runner.get("dam"),
                "damsire": runner.get("damsire"),
                "owner": runner.get("owner"),
                "comment": runner.get("comments"),
            })
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, default=DATASET_END_DATE + timedelta(days=1))
    ap.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    args = ap.parse_args()

    if args.start > args.end:
        print(f"Nothing to backfill: start {args.start} is after end {args.end}.")
        return 0

    try:
        client = RacingAPIClient()
    except RacingAPIError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    conn = connect()
    total = 0
    pages = 0
    try:
        for payload in client.results_all_pages(args.start, args.end):
            fetched_at = datetime.now(timezone.utc).isoformat()
            rows = results_to_rows(payload, fetched_at)
            n = upsert_rows(conn, rows)
            total += n
            pages += 1
            print(f"  page {pages}: {n} rows (total so far {payload.get('total')})")
    except RacingAPIError as e:
        print(f"ERROR: {e} — re-run to retry (upserts are idempotent, already-written rows are safe)", file=sys.stderr)
        return 1

    print(f"Backfill done: {total} rows across {pages} page(s) "
          f"({args.start.isoformat()} -> {args.end.isoformat()}) into live_extension.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
