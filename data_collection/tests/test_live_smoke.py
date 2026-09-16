"""
Opt-in LIVE integration test — makes real requests to api.theracingapi.com.

Skipped by default (the whole point of the mocked suite is to not need
network/credentials to run in CI or offline). Run explicitly with:

    RUN_LIVE_API_TESTS=1 pytest tests/test_live_smoke.py -v

Requires a real USERNAME/PASSWORD in the repo-root .env. Only hits the
Free-tier endpoints (racecards_free, results_today_free) — never the paid
ones, so it can't accidentally rack up usage the account isn't provisioned
for. Confirms the live schema hasn't drifted from tests/fixtures/*.json.
"""
import os

import pytest

from racingapi_client import RacingAPIClient, RacingAPIError, RacingAPIPlanError
from racingapi_client import REQUIRED_RAW_FIELDS

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_API_TESTS") != "1",
    reason="live test — set RUN_LIVE_API_TESTS=1 to run (needs real .env credentials)",
)


@pytest.fixture
def real_client():
    # The autouse no_real_env fixture in conftest.py sets fake USERNAME/PASSWORD
    # for every test (so the mocked suite never depends on real creds); undo
    # that here and force-reload the real .env (override=True, since
    # load_dotenv() by default won't clobber the fake vars already set).
    import subprocess
    from dotenv import load_dotenv
    from pathlib import Path

    here_root = Path(__file__).resolve().parent.parent.parent  # repo/worktree root
    load_dotenv(here_root / ".env", override=True)
    try:
        common_dir = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=here_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        main_root = Path(common_dir).resolve().parent
        load_dotenv(main_root / ".env", override=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return RacingAPIClient()


def test_racecards_free_returns_todays_races(real_client):
    races = real_client.racecards_free(when="today")
    assert isinstance(races, list)
    assert len(races) > 0, "expected at least one race somewhere in GB+IRE today"
    race = races[0]
    assert race.get("runners"), "expected at least one runner in the first race"


def test_racecards_free_has_every_required_pre_race_field(real_client):
    races = real_client.racecards_free(when="today")
    race = races[0]
    runner = race["runners"][0]
    combined = {**race, **runner}
    # translated field names per fetch_daily_racecards.racecard_to_rows
    translation = {"or": "ofr", "wgt": "lbs", "num": "number", "class": "race_class",
                   "sex_rest": "sex_restriction", "dist": "distance_f", "off": "off_time",
                   "hg": "headgear"}
    for field in REQUIRED_RAW_FIELDS:
        if field in ("pos", "ran"):
            continue
        api_field = translation.get(field, field)
        assert api_field in combined, f"expected API field '{api_field}' (for our '{field}') missing from live response"


def test_racecards_standard_confirmed_gated_on_free_plan(real_client):
    """Documents current account state — update/remove this test after upgrading."""
    with pytest.raises(RacingAPIPlanError):
        real_client.racecards_standard(when="today")


def test_results_paid_confirmed_gated_on_free_plan(real_client):
    import datetime
    with pytest.raises(RacingAPIPlanError):
        real_client.results(datetime.date.today() - datetime.timedelta(days=1))
