"""
Load exported settled-results JSON (the API's /v1/results shape, as saved to
files like 2026-06-26_2026-09-20_results.json) into data/live_extension.db.

Usage:
    python load_results_json.py FILE [FILE ...] [--regions GB,IRE,FR|all] [--dry-run]

- Races are de-duplicated by race_id across files (overlapping exports are fine).
- Default regions are GB,IRE,FR (what raceform.db covers). `--regions all` keeps
  every region in the files.
- Idempotent: upserts on (date, race_id, horse).
- Days that already hold pre-race racecard rows whose race_id is not one of the
  results' race_ids are MERGED instead: post-race fields (pos, btn, ovr_btn, time,
  sp, prize, comment) are copied onto the existing row, matched on (date, horse),
  falling back to the bare (unsuffixed) name for racecards fetched before the
  region-suffix fix (2026-09-16), which are renamed to the suffixed form. On
  recent days the racecard and results endpoints use the same race_id, so the
  normal upsert applies. Pre-race fields are not overwritten in the merge path.
- Only races dated after --start (default 2026-05-27, raceform.db's last day) are loaded.
- A timestamped copy of live_extension.db is made first (skipped on --dry-run).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone

from backfill_history import DATASET_END_DATE
from live_db import DB_PATH, connect, upsert_rows
from results_mapping import results_to_rows

BARE_SUFFIX = re.compile(r"\s*\([A-Za-z]{2,4}\)$")
POST_RACE_COLS = ["pos", "btn", "ovr_btn", "time", "sp", "prize", "comment"]


def load_races(paths: list[str]) -> list[dict]:
    races: dict[str, dict] = {}
    for p in paths:
        for race in json.load(open(p)).get("results", []):
            races[race["race_id"]] = race  # overlapping exports are identical
    return list(races.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--regions", default="GB,IRE,FR")
    ap.add_argument("--start", default=(DATASET_END_DATE.isoformat()),
                    help="only load races AFTER this date (default: raceform.db's last date, so history is never duplicated)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    regions = None if args.regions.lower() == "all" else {r.strip().upper() for r in args.regions.split(",")}
    races = load_races(args.files)
    fetched_at = datetime.now(timezone.utc).isoformat()
    races = [r for r in races if r["date"] > args.start]
    rows = results_to_rows({"results": races}, fetched_at, regions)
    dates = sorted({r["date"] for r in rows})
    print(f"{len(races)} races in files; {len({r['race_id'] for r in rows})} kept after region filter "
          f"({args.regions}); {len(rows)} runner rows; {dates[0]} .. {dates[-1]} ({len(dates)} days)")

    conn = connect()
    # A racecard day holds rows whose race_id is NOT one of the results' race_ids
    # (racecards use a different id scheme), so a re-run still upserts normally
    # on days this loader itself wrote.
    result_ids = {r["race_id"] for r in rows}
    racecard_days = {d for (d, rid) in conn.execute("select date, race_id from data")
                     if rid not in result_ids}
    to_merge = [r for r in rows if r["date"] in racecard_days]
    to_upsert = [r for r in rows if r["date"] not in racecard_days]
    print(f"upsert as new rows: {len(to_upsert)} ({len({r['date'] for r in to_upsert})} days); "
          f"merge into existing racecard days {sorted({r['date'] for r in to_merge})}: {len(to_merge)} rows")

    if args.dry_run:
        for r in to_upsert[:2] + to_merge[:1]:
            print({k: r[k] for k in ("date", "course", "off", "dist", "horse", "num", "pos", "sp", "wgt", "time", "or", "rpr", "ts")})
        return 0

    backup = DB_PATH.with_name(f"{DB_PATH.name}.bak-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(DB_PATH, backup)
    print(f"backup: {backup}")

    n = upsert_rows(conn, to_upsert)
    merged = ambiguous = unmatched = 0
    for r in to_merge:
        key = r["horse"]
        hits = conn.execute("select count(*) from data where date=? and horse=?", (r["date"], key)).fetchone()[0]
        if hits == 0:
            # Racecards fetched before the region-suffix fix (2026-09-16) hold bare
            # names; match on the bare name and rename the row to the suffixed form.
            key = BARE_SUFFIX.sub("", r["horse"])
            hits = conn.execute("select count(*) from data where date=? and horse=?", (r["date"], key)).fetchone()[0] if key != r["horse"] else 0
        if hits == 0:
            unmatched += 1
        elif hits > 1:
            ambiguous += 1
        else:
            sets = ", ".join(f"[{c}]=?" for c in POST_RACE_COLS)
            conn.execute(f"update data set {sets}, horse=? where date=? and horse=?",
                         [r[c] for c in POST_RACE_COLS] + [r["horse"], r["date"], key])
            merged += 1
    conn.commit()
    print(f"upserted {n} rows; merged {merged} into racecard rows; unmatched {unmatched}; ambiguous {ambiguous}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
