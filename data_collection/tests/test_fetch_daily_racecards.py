"""
Tests for racecard_to_rows() against a REAL captured /v1/racecards/free
response (tests/fixtures/racecards_free_gb.json, live 2026-09-16) — not a
hand-built fake, so a vendor field rename breaks this test rather than
silently writing NULLs into live_extension.db.
"""
from fetch_daily_racecards import racecard_to_rows
from racingapi_client import REQUIRED_RAW_FIELDS


def test_maps_real_fixture_to_expected_row_count(racecards_free_payload):
    races = racecards_free_payload["racecards"]
    rows = racecard_to_rows(races, fetched_at="2026-09-16T12:00:00Z")
    expected = sum(len(r.get("runners", [])) for r in races)
    assert expected > 0, "fixture has no runners — regenerate it"
    assert len(rows) == expected


def test_every_required_field_is_populated_for_first_runner(racecards_free_payload):
    races = racecards_free_payload["racecards"]
    rows = racecard_to_rows(races, fetched_at="2026-09-16T12:00:00Z")
    row = rows[0]
    # post-race fields must NOT be set by this script
    for post_race_field in ("pos", "sp", "rpr", "ts", "prize", "comment"):
        assert row.get(post_race_field) is None, f"{post_race_field} should be NULL pre-race"
    # every pre-race field required by features.py must at least be present
    # (empty string is a legitimate value for e.g. pattern/sex_rest — most
    # races aren't pattern races — so we check "not missing", not "truthy")
    pre_race_fields = [f for f in REQUIRED_RAW_FIELDS if f not in ("pos",)]
    for field in pre_race_fields:
        assert row.get(field) is not None, f"{field} unexpectedly missing: {row}"


def test_field_name_translation_is_correct(racecards_free_payload):
    race = racecards_free_payload["racecards"][0]
    runner = race["runners"][0]
    rows = racecard_to_rows([race], fetched_at="now")
    row = rows[0]
    assert row["or"] == runner["ofr"]
    assert row["wgt"] == runner["lbs"]
    assert row["num"] == runner["number"]
    assert row["class"] == race["race_class"]
    assert row["sex_rest"] == race["sex_restriction"]
    assert row["dist"] == race["distance_f"]


def test_ran_falls_back_to_runner_count_when_field_size_missing():
    race = {
        "date": "2026-09-16", "course": "Test", "race_id": "r1",
        "runners": [{"horse": "A"}, {"horse": "B"}],
    }
    rows = racecard_to_rows([race], fetched_at="now")
    assert rows[0]["ran"] == 2
    assert rows[1]["ran"] == 2


def test_empty_races_list_returns_empty_rows():
    assert racecard_to_rows([], fetched_at="now") == []


def test_race_with_no_runners_produces_no_rows():
    race = {"date": "2026-09-16", "course": "Test", "race_id": "r1", "runners": []}
    assert racecard_to_rows([race], fetched_at="now") == []
