import json
import sys
from pathlib import Path

import pytest

# so `import racingapi_client` etc. resolve when pytest is run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def racecards_free_payload() -> dict:
    return load_fixture("racecards_free_gb.json")


@pytest.fixture
def results_today_free_payload() -> dict:
    return load_fixture("results_today_free_gb.json")


@pytest.fixture(autouse=True)
def no_real_env(monkeypatch):
    """Every test gets deterministic fake creds unless it opts out explicitly."""
    monkeypatch.setenv("USERNAME", "test_user")
    monkeypatch.setenv("PASSWORD", "test_pass")


@pytest.fixture(autouse=True)
def reset_rate_limit_clock():
    """
    RacingAPIClient._last_request_ts is class-level/shared-across-instances
    on purpose (the vendor's rate limit is per account, not per client
    object) — reset it around every test so tests don't leak throttling
    state into each other.
    """
    import racingapi_client as rac
    rac.RacingAPIClient._last_request_ts = 0.0
    yield
    rac.RacingAPIClient._last_request_ts = 0.0
