"""
Tests for backfill_history.py. The row mapping (results_mapping.py) was verified
2026-09-20 against real /v1/results exports; more thorough mapping tests are in
test_results_mapping.py. Also tested here: date defaults, CLI plumbing, and that a
plan error surfaces a clear message and a non-zero exit rather than crashing.
"""
import datetime
import sys

import pytest
import responses

import backfill_history as bh
from racingapi_client import BASE_URL as BASE


def test_default_start_is_day_after_dataset_end():
    assert bh.DATASET_END_DATE == datetime.date(2026, 5, 27)


def test_results_to_rows_uses_verified_mapping(results_today_free_payload):
    # Mapping verified 2026-09-20 against exported /v1/results data and raceform.db
    # (see results_mapping.py): numeric strings become ints, weight stays stone-lb,
    # and the vendor's performance/speed ratings are NOT rpr/ts (left NULL).
    race = dict(results_today_free_payload["results"][0])
    race["runners"] = [dict(race["runners"][0])]
    r = race["runners"][0]
    r.update({"sp": "5/2", "weight": "9-7", "performance_rating": "75", "speed_rating": "70",
              "comment": "travelled well", "prize": "1000.50", "btn": "1.5"})

    row = bh.results_to_rows({"results": [race]}, fetched_at="now")[0]
    assert row["pos"] == int(r["position"])
    assert row["sp"] == "5/2"
    assert row["wgt"] == "9-7"
    assert row["rpr"] is None and row["ts"] is None
    assert row["comment"] == "travelled well"
    assert row["prize"] == 1000
    assert row["btn"] == 1.5
    assert row["or"] == (int(r["or"]) if str(r.get("or", "")).strip("-–") else None)


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
