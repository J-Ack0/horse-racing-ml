"""
Tests for racecard_to_rows() against a REAL captured /v1/racecards/free
response (tests/fixtures/racecards_free_gb.json, live 2026-09-16) — not a
hand-built fake, so a vendor field rename breaks this test rather than
silently writing NULLs into live_extension.db.
"""
from fetch_daily_racecards import (
    racecard_to_rows, lbs_to_wgt_str, furlongs_to_dist_str, with_region_suffix,
)
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
    from fetch_daily_racecards import lbs_to_wgt_str
    assert row["wgt"] == lbs_to_wgt_str(runner["lbs"])  # converted to stone-lbs, not raw lbs
    assert row["num"] == runner["number"]
    assert row["class"] == race["race_class"]
    assert row["sex_rest"] == race["sex_restriction"]
    from fetch_daily_racecards import furlongs_to_dist_str
    assert row["dist"] == furlongs_to_dist_str(race["distance_f"])  # converted to "Xm Yf", not raw furlongs


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


# --- weight/distance unit conversion --------------------------------------
# Regression tests: the API gives weight in plain lbs and distance in plain
# furlongs, but ml/kaggle_v2/features.py's parse_wgt()/parse_dist() expect
# raceform.db's string formats ("11-2" stone-lbs, "1m2f") and silently
# return NaN otherwise -- this was live-verified 2026-09-16 as ALL of
# wgt_lbs/dist_f (and everything derived from them) coming back NaN for
# every runner on a real inference run, caught only because it broke the
# feature output, not by mapping inspection.

def test_lbs_to_wgt_str_converts_correctly():
    assert lbs_to_wgt_str(140) == "10-0"   # 140 lbs = 10 st 0 lb
    assert lbs_to_wgt_str("140") == "10-0"  # API sends it as a string
    assert lbs_to_wgt_str(133) == "9-7"    # 133 = 9*14 + 7


def test_lbs_to_wgt_str_handles_missing():
    assert lbs_to_wgt_str(None) is None
    assert lbs_to_wgt_str("") is None
    assert lbs_to_wgt_str("not a number") is None


def test_furlongs_to_dist_str_converts_correctly():
    assert furlongs_to_dist_str("10.0") == "1m2f"   # 10f = 1 mile 2f
    assert furlongs_to_dist_str("8.0") == "1m"       # exactly 1 mile, no leftover furlongs
    assert furlongs_to_dist_str("6.0") == "6f"       # under a mile


def test_furlongs_to_dist_str_handles_missing():
    assert furlongs_to_dist_str(None) is None
    assert furlongs_to_dist_str("") is None


# --- horse-name region suffix ---------------------------------------------
# Regression: data_ext/raceform.db suffixes EVERY horse name with its region
# in parens ("Great Blasket (IRE)"), even GB-bred horses ("A Better World
# (GB)") -- confirmed live 2026-09-16 with zero exceptions in the table. The
# API's `horse` field is bare. Without matching this, features.py's
# groupby('horse') treats today's row and the horse's own historical rows as
# two unrelated entities, so h_win_rate/days_since_run/prior_rpr/etc. come
# back NaN for every live row no matter how much history is loaded -- the
# join key itself never matches. This doesn't raise or show up as an error,
# only as silently-empty features, so it's worth pinning down explicitly.

def test_with_region_suffix_appends_region():
    assert with_region_suffix("Great Blasket", "IRE") == "Great Blasket (IRE)"
    assert with_region_suffix("A Better World", "GB") == "A Better World (GB)"


def test_with_region_suffix_handles_missing_region():
    assert with_region_suffix("Great Blasket", None) == "Great Blasket"
    assert with_region_suffix("Great Blasket", "") == "Great Blasket"


def test_with_region_suffix_handles_missing_name():
    assert with_region_suffix(None, "IRE") is None


def test_racecard_to_rows_horse_name_carries_region_suffix(racecards_free_payload):
    race = racecards_free_payload["racecards"][0]
    runner = race["runners"][0]
    row = racecard_to_rows([race], fetched_at="now")[0]
    assert row["horse"] == with_region_suffix(runner["horse"], runner["region"])
    assert row["horse"] != runner["horse"], "must not write the bare name — it won't match raceform.db"


def test_racecard_to_rows_wgt_and_dist_are_parseable_by_features_py():
    """
    End-to-end regression: the strings racecard_to_rows() writes for wgt/dist
    must be exactly what features.py::parse_wgt()/parse_dist() expect, or
    every downstream feature derived from them (wgt_lbs, dist_f, log_dist,
    dist_band, all the *_z/*_rk race-relative transforms of those) goes
    silently NaN for every live row.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml" / "kaggle_v2"))
    from features import parse_wgt, parse_dist

    race = {
        "date": "2026-09-16", "course": "Test", "race_id": "r1",
        "runners": [{"horse": "A", "lbs": "140"}],
    }
    race["distance_f"] = "10.0"
    row = racecard_to_rows([race], fetched_at="now")[0]
    assert parse_wgt(row["wgt"]) == 140.0
    assert parse_dist(row["dist"]) == 10.0


def test_off_is_24h_like_raceform_db():
    from fetch_daily_racecards import off_24h
    assert off_24h({"off_time": "2:08", "off_dt": "2026-09-16T14:08:00+01:00"}) == "14:08"
    assert off_24h({"off_time": "2:08"}) == "14:08"      # no off_dt: afternoon hours are pm
    assert off_24h({"off_time": "12:30"}) == "12:30"
    assert off_24h({"off_time": "11:40"}) == "11:40"
    assert off_24h({}) is None


def test_racecard_rows_carry_24h_off(racecards_free_payload):
    races = racecards_free_payload["racecards"][:1]
    from fetch_daily_racecards import racecard_to_rows
    rows = racecard_to_rows(races, fetched_at="now")
    assert rows[0]["off"] == races[0]["off_dt"][11:16]
