"""
Thin client for The Racing API (theracingapi.com) — UK & Ireland coverage.

Auth: HTTP Basic. Credentials live in a `.env` file at the repo root
(`USERNAME=...` / `PASSWORD=...`, gitignored — see the ".env" entry added
in 56dfabf) and are loaded via `load_dotenv()` below — never hardcoded,
never logged, never printed by this module.

`USERNAME`/`PASSWORD` are generic names shared with things like your shell
login. `load_dotenv()` does NOT override a value already present in the
real environment, so if your shell happens to already export its own
USERNAME/PASSWORD those win silently. If auth ever looks wrong, check
`env | grep -E '^(USERNAME|PASSWORD)='` before assuming the .env is bad.

Endpoint paths/params/fields below are CONFIRMED LIVE against the account's
own key on 2026-09-16 (see docs/PROJECT_NOTES.md, section 4.3) — not just
vendor docs. Two important
findings from that verification:

  1. This account is on the **Free plan**. `/racecards/standard` and the
     historical `/results` endpoint both 401 with "Standard Plan required".
     `/racecards/free` and `/results/today/free` work and, for racecards,
     actually carry every pre-race field the model needs (ofr, lbs, draw,
     sire/dam/damsire, headgear, jockey, trainer, owner) — so live
     PREDICTION works today without upgrading. Historical BACKFILL
     (backfill_history.py) does NOT: `/results/today/free` only covers
     today and is missing sp/rpr/ts/prize/comment even for today. That
     needs a Standard plan subscription.
  2. Free-plan rate limit is 1 req/sec, not 5 — confirmed via a live 429
     ("Rate limit exceeded: 1 per 1 second"). Also `region_codes` takes ONE
     region per call ("gb" or "ire"), not a combined "gb+ire" (422
     "unrecognised region code, gb+ire") — REGIONS below are queried one
     at a time and merged.
"""
from __future__ import annotations

import os
import time
import json
import argparse
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


def _load_env() -> None:
    """
    Load .env from this checkout's root, and — since .env is gitignored and
    so isn't copied into git worktrees — also fall back to the main repo's
    root (found via `git rev-parse --git-common-dir`) so this works whether
    run from the main checkout or a worktree.
    """
    here_root = Path(__file__).resolve().parent.parent
    load_dotenv(here_root / ".env")
    try:
        common_dir = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=here_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        main_root = Path(common_dir).resolve().parent
        load_dotenv(main_root / ".env")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


_load_env()

BASE_URL = "https://api.theracingapi.com/v1"
REGIONS = ["gb", "ire"]  # queried one at a time and merged — see module docstring
RATE_LIMIT_PER_SEC = 1  # confirmed live for the Free plan; bump if/when upgraded

# Raw fields features.py::load_raw() needs, for reference when mapping the
# API's JSON response onto our schema:
REQUIRED_RAW_FIELDS = [
    "date", "course", "race_id", "off", "race_name", "type", "class",
    "pattern", "age_band", "sex_rest", "dist", "going", "ran", "num",
    "pos", "draw", "horse", "age", "sex", "wgt", "hg", "jockey", "trainer",
    "or", "sire", "dam", "damsire",
]


class RacingAPIError(RuntimeError):
    pass


class RacingAPIPlanError(RacingAPIError):
    """Raised when the account's plan doesn't cover the requested endpoint."""


class RacingAPIClient:
    # Class-level (shared across every instance in this process), not
    # per-instance: the rate limit is enforced by the vendor per ACCOUNT,
    # not per client object, so two RacingAPIClient()s created back-to-back
    # (e.g. in two separate script runs sharing a process, or two tests)
    # must not each think they get their own fresh 1-req/sec allowance.
    _last_request_ts = 0.0

    def __init__(self, username: str | None = None, password: str | None = None):
        self.username = username or os.environ.get("USERNAME")
        self.password = password or os.environ.get("PASSWORD")
        if not self.username or not self.password:
            raise RacingAPIError(
                "Set USERNAME and PASSWORD in the repo-root .env "
                "before using this client."
            )
        self.session = requests.Session()
        self.session.auth = (self.username, self.password)

    def _throttle(self) -> None:
        min_interval = 1.0 / RATE_LIMIT_PER_SEC
        elapsed = time.monotonic() - RacingAPIClient._last_request_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        RacingAPIClient._last_request_ts = time.monotonic()

    def _get(self, path: str, params: dict | None = None) -> Any:
        self._throttle()
        url = f"{BASE_URL}{path}"
        resp = self.session.get(url, params=params, timeout=30)
        RacingAPIClient._last_request_ts = time.monotonic()  # account for request latency too

        if resp.status_code == 200:
            return resp.json()

        try:
            detail = resp.json().get("detail") or resp.json().get("error")
        except (ValueError, AttributeError):
            detail = resp.text[:200]

        if resp.status_code == 401 and detail and "plan" in str(detail).lower():
            raise RacingAPIPlanError(f"{path}: {detail} (account plan doesn't cover this endpoint)")
        if resp.status_code == 401:
            raise RacingAPIError(f"{path}: 401 Unauthorized — check USERNAME/PASSWORD in .env ({detail})")
        if resp.status_code == 429:
            raise RacingAPIError(f"{path}: 429 rate limited ({detail}) — back off and retry.")
        if resp.status_code == 422:
            raise RacingAPIError(f"{path}: 422 validation error — {detail}")
        raise RacingAPIError(f"{path}: HTTP {resp.status_code} — {detail}")

    # --- Racecards (pre-race, for live prediction) ----------------------
    # Free-plan endpoint. Carries every pre-race field REQUIRED_RAW_FIELDS
    # needs (verified live 2026-09-16) — no plan upgrade needed for this.

    def racecards_free(self, when: str = "today") -> list[dict]:
        """
        Racecards for "today" or "tomorrow", merged across REGIONS. Each
        call only takes one region, so this makes len(REGIONS) requests.

        NOTE: /racecards/free actually only supports day="today" — "tomorrow"
        is accepted as a param by this client but not guaranteed to return
        anything until verified; the pre-race evening-fetch flow may need
        to fall back to same-day-only until a Standard plan is available
        (see docs/PROJECT_NOTES.md, section 4.3).
        """
        races: list[dict] = []
        for region in REGIONS:
            payload = self._get("/racecards/free", params={"region_codes": region, "day": when})
            races.extend(payload.get("racecards", []))
        return races

    def racecards_standard(self, when: str = "today") -> list[dict]:
        """Paid-plan racecards (richer fields incl. bookmaker odds). Raises RacingAPIPlanError on Free."""
        if when not in ("today", "tomorrow"):
            raise ValueError('racecards_standard() day must be "today" or "tomorrow"')
        races: list[dict] = []
        for region in REGIONS:
            payload = self._get("/racecards/standard", params={"day": when, "region_codes": region})
            races.extend(payload.get("racecards", []))
        return races

    # --- Results (post-race, for backfilling training history) ---------

    def results_today_free(self) -> list[dict]:
        """
        Free-plan today's results. Missing sp/rpr/ts/prize/comment (verified
        live 2026-09-16) — usable for a coarse position-only signal, NOT a
        substitute for the Standard-plan historical backfill below.
        """
        races: list[dict] = []
        for region in REGIONS:
            payload = self._get("/results/today/free", params={"region": region})
            races.extend(payload.get("results", []))
        return races

    def results(self, start: date, end: date | None = None, skip: int = 0) -> Any:
        """
        Paid-plan historical results for [start, end] (inclusive).
        Raises RacingAPIPlanError on the Free plan (confirmed live).
        One page (limit=500); paginate by passing `skip` — the response's
        "total" field tells you when you've read everything.
        """
        end = end or start
        return self._get(
            "/results",
            params={
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "region": "gb,ire",
                "limit": 500,
                "skip": skip,
            },
        )

    def results_all_pages(self, start: date, end: date | None = None):
        """Generator yielding every page's payload until `total` is exhausted."""
        skip = 0
        while True:
            payload = self.results(start, end, skip=skip)
            yield payload
            total = payload.get("total", 0)
            skip += 500
            if skip >= total:
                break


def _probe(client: RacingAPIClient) -> None:
    """Dump live racecard + results data (free tier) so field names can be checked/mapped."""
    races = client.racecards_free(when="today")
    print(f"racecards_free: {len(races)} races")
    if races:
        print(json.dumps(races[0], indent=2)[:3000])
    print("\n--- REQUIRED_RAW_FIELDS to locate in the above ---")
    print(REQUIRED_RAW_FIELDS)

    print("\n--- trying /racecards/standard (expect RacingAPIPlanError on Free) ---")
    try:
        client.racecards_standard(when="today")
        print("Standard plan racecards worked — upgrade already in effect!")
    except RacingAPIPlanError as e:
        print(f"As expected on Free plan: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="fetch today's racecard and print it")
    args = ap.parse_args()
    c = RacingAPIClient()
    if args.probe:
        _probe(c)
