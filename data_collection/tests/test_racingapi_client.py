"""
Unit tests for racingapi_client.py — auth, error handling, rate limiting,
region-merging, and pagination. All HTTP is mocked (no live calls here);
see tests/test_live_smoke.py for the opt-in live-API check.
"""
import time

import pytest
import responses

import racingapi_client as rac
from racingapi_client import RacingAPIClient, RacingAPIError, RacingAPIPlanError

BASE = rac.BASE_URL


# --- Construction / credentials -----------------------------------------

def test_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("USERNAME", raising=False)
    monkeypatch.delenv("PASSWORD", raising=False)
    with pytest.raises(RacingAPIError, match="USERNAME and PASSWORD"):
        RacingAPIClient()


def test_credentials_from_env(monkeypatch):
    monkeypatch.setenv("USERNAME", "alice")
    monkeypatch.setenv("PASSWORD", "s3cret")
    c = RacingAPIClient()
    assert c.session.auth == ("alice", "s3cret")


def test_credentials_never_leak_into_default_repr():
    # RacingAPIClient doesn't define __repr__, so this is Python's default
    # object repr (module + id) — confirms we haven't accidentally added a
    # __repr__/__str__ that would print credentials in a log/traceback.
    c = RacingAPIClient(username="alice", password="s3cret")
    assert "s3cret" not in repr(c)


def test_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv("USERNAME", "envuser")
    monkeypatch.setenv("PASSWORD", "envpass")
    c = RacingAPIClient(username="explicit", password="explicitpass")
    assert c.session.auth == ("explicit", "explicitpass")


# --- Error mapping --------------------------------------------------------

@responses.activate
def test_401_with_plan_detail_raises_plan_error():
    responses.add(
        responses.GET, f"{BASE}/racecards/standard",
        json={"detail": "Standard Plan required"}, status=401,
    )
    c = RacingAPIClient(username="u", password="p")
    with pytest.raises(RacingAPIPlanError, match="Standard Plan required"):
        c._get("/racecards/standard", params={"day": "today"})


@responses.activate
def test_401_without_plan_detail_raises_generic_error():
    responses.add(
        responses.GET, f"{BASE}/racecards/standard",
        json={"detail": "Invalid credentials"}, status=401,
    )
    c = RacingAPIClient(username="u", password="p")
    with pytest.raises(RacingAPIError) as exc_info:
        c._get("/racecards/standard")
    assert not isinstance(exc_info.value, RacingAPIPlanError)
    assert "Invalid credentials" in str(exc_info.value)


@responses.activate
def test_429_raises_with_detail():
    responses.add(
        responses.GET, f"{BASE}/racecards/free",
        json={"error": "Rate limit exceeded: 1 per 1 second"}, status=429,
    )
    c = RacingAPIClient(username="u", password="p")
    with pytest.raises(RacingAPIError, match="429"):
        c._get("/racecards/free")


@responses.activate
def test_422_raises_with_detail():
    responses.add(
        responses.GET, f"{BASE}/racecards/free",
        json={"detail": "Validation error - unrecognised region code, gb+ire"}, status=422,
    )
    c = RacingAPIClient(username="u", password="p")
    with pytest.raises(RacingAPIError, match="unrecognised region code"):
        c._get("/racecards/free")


@responses.activate
def test_non_json_error_body_does_not_crash():
    responses.add(
        responses.GET, f"{BASE}/racecards/free",
        body="<html>502 Bad Gateway</html>", status=502,
    )
    c = RacingAPIClient(username="u", password="p")
    with pytest.raises(RacingAPIError, match="502"):
        c._get("/racecards/free")


@responses.activate
def test_200_returns_parsed_json():
    responses.add(
        responses.GET, f"{BASE}/racecards/free",
        json={"racecards": [], "total": 0}, status=200,
    )
    c = RacingAPIClient(username="u", password="p")
    assert c._get("/racecards/free") == {"racecards": [], "total": 0}


# --- Rate limiting ---------------------------------------------------------

@responses.activate
def test_throttle_enforces_minimum_interval(monkeypatch):
    responses.add(responses.GET, f"{BASE}/racecards/free", json={"racecards": []}, status=200)
    responses.add(responses.GET, f"{BASE}/racecards/free", json={"racecards": []}, status=200)

    sleeps = []
    monkeypatch.setattr(rac.time, "sleep", lambda s: sleeps.append(s))

    c = RacingAPIClient(username="u", password="p")
    RacingAPIClient._last_request_ts = time.monotonic()  # simulate a request having "just happened"
    c._get("/racecards/free")

    assert sleeps, "expected a throttle sleep before the second-in-a-row request"
    assert sleeps[0] <= 1.0 / rac.RATE_LIMIT_PER_SEC + 1e-6


@responses.activate
def test_rate_limit_clock_is_shared_across_instances(monkeypatch):
    """
    Regression test: the throttle clock must be class-level, not per-instance
    — two separate RacingAPIClient() objects hitting the API back-to-back
    (as happens across two script invocations sharing a process, or in
    back-to-back test functions) share the vendor's per-account rate limit
    and must throttle against each other, not each get their own fresh
    1-req/sec allowance.
    """
    responses.add(responses.GET, f"{BASE}/racecards/free", json={"racecards": []}, status=200)
    responses.add(responses.GET, f"{BASE}/racecards/free", json={"racecards": []}, status=200)
    sleeps = []
    monkeypatch.setattr(rac.time, "sleep", lambda s: sleeps.append(s))

    c1 = RacingAPIClient(username="u", password="p")
    c1._get("/racecards/free")
    c2 = RacingAPIClient(username="u", password="p")  # brand new instance
    c2._get("/racecards/free")

    assert sleeps, "second client's first request should still be throttled against the first client's request"


@responses.activate
def test_no_throttle_sleep_when_enough_time_elapsed(monkeypatch):
    responses.add(responses.GET, f"{BASE}/racecards/free", json={"racecards": []}, status=200)
    sleeps = []
    monkeypatch.setattr(rac.time, "sleep", lambda s: sleeps.append(s))

    c = RacingAPIClient(username="u", password="p")
    RacingAPIClient._last_request_ts = 0.0  # "ages ago"
    c._get("/racecards/free")

    assert sleeps == []


# --- Region merging --------------------------------------------------------

@responses.activate
def test_racecards_free_merges_all_regions():
    responses.add(
        responses.GET, f"{BASE}/racecards/free",
        json={"racecards": [{"race_id": "gb_1"}]}, status=200,
        match=[responses.matchers.query_param_matcher({"region_codes": "gb", "day": "today"})],
    )
    responses.add(
        responses.GET, f"{BASE}/racecards/free",
        json={"racecards": [{"race_id": "ire_1"}]}, status=200,
        match=[responses.matchers.query_param_matcher({"region_codes": "ire", "day": "today"})],
    )
    c = RacingAPIClient(username="u", password="p")
    races = c.racecards_free(when="today")
    assert [r["race_id"] for r in races] == ["gb_1", "ire_1"]


def test_racecards_standard_rejects_bad_day():
    c = RacingAPIClient(username="u", password="p")
    with pytest.raises(ValueError):
        c.racecards_standard(when="yesterday")


# --- Pagination --------------------------------------------------------

@responses.activate
def test_results_all_pages_stops_at_total(monkeypatch):
    monkeypatch.setattr(rac.time, "sleep", lambda s: None)
    import datetime
    for region in ("gb", "ire"):
        total = 120 if region == "gb" else 30
        responses.add(
            responses.GET, f"{BASE}/results",
            json={"results": [{"race_id": f"{region}{i}"} for i in range(min(100, total))], "total": total}, status=200,
            match=[responses.matchers.query_param_matcher({
                "start_date": "2026-01-01", "end_date": "2026-01-01", "region": region, "limit": "100", "skip": "0"})],
        )
    responses.add(
        responses.GET, f"{BASE}/results",
        json={"results": [{"race_id": f"gb{i}"} for i in range(100, 120)], "total": 120}, status=200,
        match=[responses.matchers.query_param_matcher({
            "start_date": "2026-01-01", "end_date": "2026-01-01", "region": "gb", "limit": "100", "skip": "100"})],
    )
    c = RacingAPIClient(username="u", password="p")
    pages = list(c.results_all_pages(datetime.date(2026, 1, 1)))
    # gb needs two pages (120 races, 100 per page), ire one page
    assert [len(p["results"]) for p in pages] == [100, 20, 30]


@responses.activate
def test_results_all_pages_single_page_when_total_fits(monkeypatch):
    monkeypatch.setattr(rac.time, "sleep", lambda s: None)
    responses.add(
        responses.GET, f"{BASE}/results",
        json={"results": [{"race_id": "r1"}], "total": 1}, status=200,
    )
    import datetime
    c = RacingAPIClient(username="u", password="p")
    pages = list(c.results_all_pages(datetime.date(2026, 1, 1)))
    assert len(pages) == 2   # one page per region (gb, ire)


@responses.activate
def test_results_propagates_plan_error_mid_pagination(monkeypatch):
    monkeypatch.setattr(rac.time, "sleep", lambda s: None)
    responses.add(
        responses.GET, f"{BASE}/results",
        json={"detail": "Standard Plan required"}, status=401,
    )
    import datetime
    c = RacingAPIClient(username="u", password="p")
    with pytest.raises(RacingAPIPlanError):
        list(c.results_all_pages(datetime.date(2026, 1, 1)))
