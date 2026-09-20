"""
Daily job: pull racecards for live prediction.

Two intended run times (see docs/PROJECT_NOTES.md, section 4.2):
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


def off_24h(race: dict) -> str | None:
    """
    "14:08" from the racecard's off_dt (local ISO time). The API's own `off_time` is a 12h clock with
    no am/pm ("2:08"), while raceform.db stores 24h ("14:08") and features.py's `off_hour` is the
    leading number, so passing "2:08" through made every afternoon race look like 2am (found
    2026-09-20). Fallback without off_dt: UK/IRE racing runs about 11:00-21:00, so hours 1-9 are pm.
    """
    dt = race.get("off_dt") or ""
    if len(dt) >= 16 and dt[10] == "T":
        return dt[11:16]
    t = race.get("off_time")
    if not t or ":" not in str(t):
        return t
    h, _, m = str(t).partition(":")
    try:
        hour = int(h)
    except ValueError:
        return t
    return f"{hour + 12 if 1 <= hour <= 9 else hour:02d}:{m}"


def lbs_to_wgt_str(lbs) -> str | None:
    """
    140 (lbs, as returned by the API's `lbs` field) -> "10-0" (stone-lbs,
    the format ml/kaggle_v2/features.py::parse_wgt() expects, matching how
    raceform.db stores it). Without this, parse_wgt() sees a bare number
    with no "-" and silently returns NaN for every live row — caught when
    inference.py reported wgt_lbs as all-NaN for today's racecard.
    """
    if lbs in (None, ""):
        return None
    try:
        lbs = float(lbs)
    except (TypeError, ValueError):
        return None
    st, rem = divmod(lbs, 14)
    return f"{int(st)}-{rem:g}"


def furlongs_to_dist_str(distance_f) -> str | None:
    """
    "10.0" (furlongs, as returned by the API's `distance_f` field) -> "1m2f"
    (the format parse_dist() expects). Same failure mode as lbs_to_wgt_str:
    parse_dist() needs an 'm'/'f'-suffixed string, not a bare furlongs number.
    """
    if distance_f in (None, ""):
        return None
    try:
        f = float(distance_f)
    except (TypeError, ValueError):
        return None
    miles, rem = divmod(f, 8)
    if miles > 0:
        return f"{int(miles)}m" + (f"{rem:g}f" if rem > 0 else "")
    return f"{rem:g}f"


def with_region_suffix(name, region) -> str | None:
    """
    "Great Blasket" + "IRE" -> "Great Blasket (IRE)" -- data_ext/raceform.db
    suffixes EVERY horse name with its region in parens, no exceptions (even
    GB-bred horses get "(GB)", verified live 2026-09-16), but the API's
    `horse` field is bare. Without this, features.py::build()'s
    groupby('horse') treats today's row and that horse's own history as two
    different entities -- every prior_*/h_* feature (h_win_rate,
    days_since_run, prior_rpr, ...) silently comes back NaN for every live
    row regardless of how much history is loaded, since the join key itself
    never matches. Caught only by inspecting the actual feature output, not
    by any test — the mismatch doesn't raise, it just produces zeros.
    """
    if not name:
        return name
    if not region:
        return name
    return f"{name} ({region})"


def racecard_to_rows(races: list[dict], fetched_at: str) -> list[dict]:
    """
    Map a list of races from /v1/racecards/free (racingapi_client.racecards_free())
    onto our `data` table row shape.

    Field names below are LIVE-CONFIRMED against the account's own key on
    2026-09-16 (see docs/PROJECT_NOTES.md, section 4.3). Deltas from our column names: API `ofr` ->
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
            "off": off_24h(race),
            "race_name": race.get("race_name"),
            "type": race.get("type"),
            "class": race.get("race_class"),
            "pattern": race.get("pattern"),
            "rating_band": race.get("rating_band"),
            "age_band": race.get("age_band"),
            "sex_rest": race.get("sex_restriction"),
            "dist": furlongs_to_dist_str(race.get("distance_f")),
            "going": race.get("going"),
            "ran": race.get("field_size") or len(race.get("runners", [])),
            "fetched_at": fetched_at,
        }
        for runner in race.get("runners", []):
            row = dict(base)
            row.update({
                "num": runner.get("number"),
                "draw": runner.get("draw"),
                "horse": with_region_suffix(runner.get("horse"), runner.get("region")),
                "age": runner.get("age"),
                "sex": runner.get("sex"),
                "wgt": lbs_to_wgt_str(runner.get("lbs")),
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
