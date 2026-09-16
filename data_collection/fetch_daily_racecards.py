"""
Daily job: pull racecards for live prediction.

Two intended run times (see docs/2026-09-14_live_data_pipeline_plan.md):
  1. Evening (e.g. 18:00) — pull TOMORROW's racecard. Fields are stable this
     far out (declarations, entries) and this is what next-day feature
     building/prediction runs against.
  2. Pre-race (e.g. 60-90 min before first race) — re-pull TODAY's racecard
     to pick up non-runners / declaration changes / late jockey changes.

This script ONLY writes pre-race fields (see racingapi_client.REQUIRED_RAW_FIELDS)
— no result/pos/sp/rpr/ts data, since those don't exist yet. It must never be
used to backfill history; use backfill_history.py for that.

Usage:
    python fetch_daily_racecards.py --day tomorrow
    python fetch_daily_racecards.py --day today
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta, datetime, timezone


from racingapi_client import RacingAPIClient, RacingAPIError
from live_db import connect, upsert_rows


def racecard_to_rows(races: list[dict], fetched_at: str) -> list[dict]:
    """
    Map a list of races from /v1/racecards/free (racingapi_client.racecards_free())
    onto our `data` table row shape.

    Field names below are LIVE-CONFIRMED against the account's own key on
    2026-09-16 (see docs/2026-09-14_live_data_pipeline_plan.md,
    "Live-verified 2026-09-16"). Deltas from our column names: API `ofr` ->
    our `or`, API `lbs` -> our `wgt`, API `race_class` -> our `class`, API
    `distance_f` -> our `dist` (furlongs-as-string; the free tier has no
    "6f210y"-style distance string, only distance_f). Only pre-race fields
    are populated; pos/sp/rpr/ts/prize/comment are left NULL since they
    don't exist until after the race.
    """
    rows: list[dict] = []
    for race in races:
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
            "dist": race.get("distance_f"),
            "going": race.get("going"),
            "ran": race.get("field_size") or len(race.get("runners", [])),
            "fetched_at": fetched_at,
        }
        for runner in race.get("runners", []):
            row = dict(base)
            row.update({
                "num": runner.get("number"),
                "draw": runner.get("draw"),
                "horse": runner.get("horse"),
                "age": runner.get("age"),
                "sex": runner.get("sex"),
                "wgt": runner.get("lbs"),
                "hg": runner.get("headgear"),
                "jockey": runner.get("jockey"),
                "trainer": runner.get("trainer"),
                "or": runner.get("ofr"),
                "sire": runner.get("sire"),
                "dam": runner.get("dam"),
                "damsire": runner.get("damsire"),
                "owner": runner.get("owner"),
            })
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", choices=["today", "tomorrow"], default="tomorrow")
    args = ap.parse_args()

    target = date.today() if args.day == "today" else date.today() + timedelta(days=1)
    fetched_at = datetime.now(timezone.utc).isoformat()

    try:
        client = RacingAPIClient()
        races = client.racecards_free(when=args.day)
    except RacingAPIError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    rows = racecard_to_rows(races, fetched_at)
    if not rows:
        print(f"No racecard rows returned for {target.isoformat()} — nothing written.")
        return 0

    conn = connect()
    n = upsert_rows(conn, rows)
    print(f"Upserted {n} runner rows for {target.isoformat()} into live_extension.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
