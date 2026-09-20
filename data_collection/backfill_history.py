"""
One-off (then daily) backfill: pull SETTLED RESULTS day-by-day and write them
into live_extension.db, so the gap between the static Kaggle export's last
date and "yesterday" gets filled in for rolling/entity feature history.

The static export (data/raceform.db) currently ends 2026-05-27 (verified via
`SELECT MIN(date), MAX(date) FROM data` — see
docs/PROJECT_NOTES.md, section 2.1). Default start is the day after
that; override with --start if the export gets refreshed later.

This writes POST-race fields (pos, sp, rpr, ts, prize) which is fine here —
this script backfills TRAINING history, it is never used for the live
pre-race feature build (fetch_daily_racecards.py does that, pre-race fields
only). Keeping the two scripts separate is deliberate: it makes it
structurally impossible to leak a result into a same-day pre-race feature.

The API's /v1/results endpoint takes a start_date/end_date range directly
(not one call per day) and paginates at limit=500 rows/page.

⚠️ LIVE-CONFIRMED 2026-09-16: this endpoint 401s with "Standard Plan
required" on the account's current Free plan — see
docs/PROJECT_NOTES.md, section 4.3 (live-verified 2026-09-16).
The account is now on the Standard plan. The row mapping lives in
results_mapping.py and was verified 2026-09-20 against exported
/v1/results data and raceform.db (see that module's docstring).

Usage:
    python backfill_history.py                      # day after the last day with results -> yesterday
    python backfill_history.py --start 2026-06-01 --end 2026-06-30
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta, datetime, timezone

from racingapi_client import RacingAPIClient, RacingAPIError, RacingAPIPlanError
from live_db import connect, upsert_rows
from results_mapping import results_to_rows  # verified mapping, see results_mapping.py

DATASET_END_DATE = date(2026, 5, 27)  # last date present in data/raceform.db


def default_start() -> date:
    """Day after the latest day that already has results in live_extension.db."""
    conn = connect()
    row = conn.execute("SELECT MAX(date) FROM data WHERE pos IS NOT NULL").fetchone()
    conn.close()
    return date.fromisoformat(row[0]) + timedelta(days=1) if row and row[0] else DATASET_END_DATE + timedelta(days=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, default=None,
                    help="default: the day after the latest day with results in live_extension.db")
    ap.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    args = ap.parse_args()
    if args.start is None:
        args.start = default_start()

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
    except RacingAPIPlanError as e:
        print(f"ERROR: {e}\nHistorical backfill needs a Standard plan subscription "
              f"(theracingapi.com) — the Free plan only covers today's racecards/results.",
              file=sys.stderr)
        return 1
    except RacingAPIError as e:
        print(f"ERROR: {e} — re-run to retry (upserts are idempotent, already-written rows are safe)", file=sys.stderr)
        return 1

    print(f"Backfill done: {total} rows across {pages} page(s) "
          f"({args.start.isoformat()} -> {args.end.isoformat()}) into live_extension.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
