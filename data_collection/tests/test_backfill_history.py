"""
Tests for backfill_history.py. results_to_rows()'s exact field mapping is
UNVERIFIED against a real paid response (the account is on the Free plan —
see racingapi_client.py's module docstring), so these tests pin the
*documented/assumed* shape and are expected to need updating the moment a
Standard-plan response is captured. What IS fully tested here: date
defaults, CLI plumbing, and — critically — that a plan error surfaces a
clear message and a non-zero exit rather than crashing or silently no-oping.
"""
import datetime
import sys

import pytest
import responses

import backfill_history as bh
from racingapi_client import BASE_URL as BASE


def test_default_start_is_day_after_dataset_end():
    assert bh.DATASET_END_DATE == datetime.date(2026, 5, 27)


def test_results_to_rows_maps_assumed_field_names(results_today_free_payload):
    # NOTE: results_today_free lacks sp/rpr/ts/prize/comment entirely (confirmed
    # live) so this fixture only exercises the fields it does share with the
    # assumed paid /results shape (position, weight_lbs is NOT in free -> weight
    # only). We test the mapping function's *logic* here, not full field coverage;
    # full coverage needs a paid-plan fixture (see module docstring).
    race = dict(results_today_free_payload["results"][0])
    race["runners"] = [dict(race["runners"][0])]
    r = race["runners"][0]
    r["sp_dec"] = 2.5
    r["weight_lbs"] = r.get("weight_lbs", "140")
    r["performance_rating"] = 75
    r["speed_rating"] = 70
    r["comments"] = "travelled well"
    r["prize"] = 1000
    r["btn"] = "1.5"

    rows = bh.results_to_rows({"results": [race]}, fetched_at="now")
    row = rows[0]
    assert row["pos"] == r["position"]
    assert row["sp"] == 2.5
    assert row["rpr"] == 75
    assert row["ts"] == 70
    assert row["comment"] == "travelled well"
    assert row["prize"] == 1000
    # results/today/free uses key "or" directly (confirmed live 2026-09-16,
    # unlike racecards/free which uses "ofr" for the same field)
    assert row["or"] == r["or"]


def test_results_to_rows_empty_payload():
    assert bh.results_to_rows({"results": []}, fetched_at="now") == []


def test_results_to_rows_race_with_no_runners():
    race = {"date": "2026-09-01", "course": "Test", "race_id": "r1", "runners": []}
    assert bh.results_to_rows({"results": [race]}, fetched_at="now") == []


# --- main() plan-gating behavior -----------------------------------------

@responses.activate
def test_main_reports_plan_error_and_exits_nonzero(monkeypatch, capsys):
    responses.add(
        responses.GET, f"{BASE}/results",
        json={"detail": "Standard Plan required"}, status=401,
    )
    monkeypatch.setattr(sys, "argv", ["backfill_history.py", "--start", "2026-09-01", "--end", "2026-09-01"])
    rc = bh.main()
    assert rc == 1
    captured = capsys.readouterr()
    assert "Standard plan" in captured.err or "Standard Plan" in captured.err


def test_main_noop_when_start_after_end(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["backfill_history.py", "--start", "2026-09-05", "--end", "2026-09-01"])
    rc = bh.main()
    assert rc == 0
    assert "Nothing to backfill" in capsys.readouterr().out


def test_main_fails_cleanly_without_credentials(monkeypatch, capsys):
    monkeypatch.delenv("USERNAME", raising=False)
    monkeypatch.delenv("PASSWORD", raising=False)
    monkeypatch.setattr(sys, "argv", ["backfill_history.py", "--start", "2026-09-01", "--end", "2026-09-01"])
    rc = bh.main()
    assert rc == 1
    assert "USERNAME and PASSWORD" in capsys.readouterr().err
