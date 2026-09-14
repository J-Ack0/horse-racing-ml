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

Usage:
    python backfill_history.py                      # 2026-05-28 -> yesterday
    python backfill_history.py --start 2026-06-01 --end 2026-06-30
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta, datetime, timezone

from racingapi_client import RacingAPIClient, RacingAPIError
from live_db import connect, upsert_rows

DATASET_END_DATE = date(2026, 5, 27)  # last date present in data/raceform.db


def results_to_rows(payload: dict, fetched_at: str) -> list[dict]:
    """
    Map one /results response onto our `data` table row shape.

    ASSUMPTION: payload shape mirrors racecards
    ({"results": [{race fields..., "runners": [...]}]}) with runner-level
    finishing fields added (position, sp, ratings). VERIFY with a probe
    result response and adjust key names before relying on this.
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
            "ran": len(race.get("runners", [])),
            "fetched_at": fetched_at,
        }
        for runner in race.get("runners", []):
            row = dict(base)
            row.update({
                "num": runner.get("number"),
                "draw": runner.get("draw"),
                "pos": runner.get("position"),
                "ovr_btn": runner.get("overall_beaten_distance"),
                "btn": runner.get("beaten_distance"),
                "horse": runner.get("horse"),
                "age": runner.get("age"),
                "sex": runner.get("sex"),
                "wgt": runner.get("weight"),
                "hg": runner.get("headgear"),
                "time": runner.get("time"),
                "sp": runner.get("starting_price"),
                "jockey": runner.get("jockey"),
                "trainer": runner.get("trainer"),
                "prize": runner.get("prize"),
                "or": runner.get("official_rating"),
                "rpr": runner.get("rpr"),
                "ts": runner.get("ts"),
                "sire": runner.get("sire"),
                "dam": runner.get("dam"),
                "damsire": runner.get("damsire"),
                "owner": runner.get("owner"),
                "comment": runner.get("comment"),
            })
            rows.append(row)
    return rows


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


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
    for d in daterange(args.start, args.end):
        fetched_at = datetime.now(timezone.utc).isoformat()
        try:
            payload = client.results(d)
        except RacingAPIError as e:
            print(f"  {d.isoformat()}: ERROR {e} — skipping, re-run later to retry this date", file=sys.stderr)
            continue
        rows = results_to_rows(payload, fetched_at)
        n = upsert_rows(conn, rows)
        total += n
        print(f"  {d.isoformat()}: {n} rows")

    print(f"Backfill done: {total} rows across {(args.end - args.start).days + 1} days "
          f"({args.start.isoformat()} -> {args.end.isoformat()}) into live_extension.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
