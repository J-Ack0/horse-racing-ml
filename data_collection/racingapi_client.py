"""
Thin client for The Racing API (theracingapi.com) — UK & Ireland coverage.

Auth: HTTP Basic (username/password issued on the dashboard after subscribing),
read from environment variables so credentials never touch the repo:

    RACING_API_USERNAME
    RACING_API_PASSWORD

Put these in a .env file inside your venv (or export them in the venv's
activate script) — do NOT commit them. Load with, e.g.:

    source venv/bin/activate
    export $(grep -v '^#' venv/.env | xargs)   # or use python-dotenv

ASSUMPTIONS TO VERIFY (docs are gated behind signup, could not be confirmed
without an account — see docs/2026-09-14_live_data_pipeline_plan.md):
  - base URL: https://api.theracingapi.com/v1
  - endpoint paths below (racecards, results)
  - exact field names in the JSON response
Once you have a key, run `python data_collection/racingapi_client.py --probe`
to dump one live racecard response and confirm/fix the paths & field map.
"""
from __future__ import annotations

import os
import time
import json
import argparse
from datetime import date, datetime
from typing import Any

import requests

BASE_URL = "https://api.theracingapi.com/v1"
REGION = "gb+ire"  # GB + Ireland, matches the UK+IRE training data
RATE_LIMIT_PER_SEC = 5  # per theracingapi.com docs, default plan limit

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


class RacingAPIClient:
    def __init__(self, username: str | None = None, password: str | None = None):
        self.username = username or os.environ.get("RACING_API_USERNAME")
        self.password = password or os.environ.get("RACING_API_PASSWORD")
        if not self.username or not self.password:
            raise RacingAPIError(
                "Set RACING_API_USERNAME and RACING_API_PASSWORD in the environment "
                "(e.g. a .env file in your venv) before using this client."
            )
        self.session = requests.Session()
        self.session.auth = (self.username, self.password)
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        min_interval = 1.0 / RATE_LIMIT_PER_SEC
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_ts = time.monotonic()

    def _get(self, path: str, params: dict | None = None) -> Any:
        self._throttle()
        url = f"{BASE_URL}{path}"
        resp = self.session.get(url, params=params, timeout=30)
        if resp.status_code == 401:
            raise RacingAPIError("401 Unauthorized — check RACING_API_USERNAME/PASSWORD.")
        if resp.status_code == 429:
            raise RacingAPIError("429 rate limited — back off and retry.")
        resp.raise_for_status()
        return resp.json()

    # --- Racecards (pre-race, for live prediction) ---------------------

    def racecards(self, day: date | None = None) -> Any:
        """Racecards for a given day (default: today). Pre-race fields only."""
        day = day or date.today()
        return self._get(
            "/racecards/standard",
            params={"date": day.isoformat(), "region_codes": REGION},
        )

    # --- Results (post-race, for backfilling training history) ---------

    def results(self, day: date) -> Any:
        """Settled results for a single past day."""
        return self._get(
            "/results",
            params={"date": day.isoformat(), "region_codes": REGION},
        )


def _probe(client: RacingAPIClient) -> None:
    """Dump one live racecard response so field names can be checked/mapped."""
    data = client.racecards()
    print(json.dumps(data, indent=2)[:4000])
    print("\n--- REQUIRED_RAW_FIELDS to locate in the above ---")
    print(REQUIRED_RAW_FIELDS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="fetch today's racecard and print it")
    args = ap.parse_args()
    c = RacingAPIClient()
    if args.probe:
        _probe(c)
