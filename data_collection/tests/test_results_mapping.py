"""results_to_rows() against the verified /v1/results shape, and the loader's merge logic."""
import json
import sqlite3

import pytest

import live_db
import load_results_json as lrj
from results_mapping import results_to_rows

RACE = {
    "race_id": "rac_1", "date": "2026-06-01", "region": "GB", "course": "Kempton (AW)",
    "off": "9:00", "off_dt": "2026-06-01T21:00:00+01:00", "race_name": "Test Stakes",
    "type": "Flat", "class": "Class 4", "pattern": "", "rating_band": "0-80",
    "age_band": "4yo+", "sex_rest": "", "dist": "2m13y", "going": "Standard To Slow",
    "runners": [
        {"horse": "Abundant (IRE)", "number": "10", "position": "1", "draw": "3", "btn": "0",
         "ovr_btn": "0", "age": "8", "sex": "G", "weight": "8-13", "weight_lbs": "125",
         "headgear": "t", "time": "3:5.14", "sp": "11/4F", "sp_dec": "3.75", "or": "67",
         "prize": "6280.80", "jockey": "A B", "trainer": "C D", "sire": "S (GB)", "dam": "D (GB)",
         "damsire": "DS", "owner": "O", "comment": "won well",
         "performance_rating": "73", "speed_rating": "75", "rpr": "", "tsr": ""},
        {"horse": "Faller (GB)", "number": "2", "position": "F", "draw": "", "btn": "", "ovr_btn": "",
         "weight": "9-0", "time": "", "sp": "20/1", "or": "", "prize": "", "comment": ""},
    ],
}


def test_mapping_matches_history_formats():
    a, b = results_to_rows({"results": [RACE]}, "now")
    assert a["off"] == "21:00"                # 24h from off_dt, not the 12h "9:00"
    assert a["dist"] == "2m"                  # yardage dropped like raceform
    assert a["wgt"] == "8-13"                 # stone-lb, not weight_lbs
    assert a["time"] == "3:05.14"             # zero-padded seconds
    assert a["sp"] == "11/4F"                 # fractional string, like history
    assert (a["pos"], a["num"], a["draw"], a["or"], a["prize"]) == (1, 10, 3, 67, 6280)
    assert a["rpr"] is None and a["ts"] is None   # vendor ratings are not rpr/ts
    assert a["ran"] == 2
    assert b["pos"] == "F" and b["draw"] is None and b["or"] is None and b["time"] is None


def test_region_filter():
    assert results_to_rows({"results": [RACE]}, "now", {"FR"}) == []
    assert len(results_to_rows({"results": [RACE]}, "now", {"GB"})) == 2


def test_loader_merges_racecard_days_and_upserts_new_days(tmp_path, monkeypatch):
    db = tmp_path / "live.db"
    monkeypatch.setattr(live_db, "DB_PATH", db)
    monkeypatch.setattr(lrj, "DB_PATH", db)
    conn = live_db.connect()
    # existing pre-race racecard row for a day the results file also covers
    live_db.upsert_rows(conn, [{"date": "2026-09-16", "race_id": "rac_card", "horse": "Abundant (IRE)",
                                "course": "Kempton (AW)", "or": 70}])
    conn.close()
    race_new = dict(RACE, race_id="rac_2", date="2026-06-02")
    race_merge = dict(RACE, race_id="rac_3", date="2026-09-16")
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"results": [race_new, race_merge]}))
    monkeypatch.setattr("sys.argv", ["x", str(f)])
    assert lrj.main() == 0
    con = sqlite3.connect(db)
    assert con.execute("select count(*) from data where date='2026-06-02'").fetchone()[0] == 2
    # racecard day: no duplicate row, post-race fields merged, pre-race `or` kept
    rows = con.execute("select race_id, pos, sp, [or] from data where date='2026-09-16'").fetchall()
    assert rows == [("rac_card", 1, "11/4F", 70)]


def test_loader_matches_bare_names_from_pre_suffix_racecards(tmp_path, monkeypatch):
    db = tmp_path / "live.db"
    monkeypatch.setattr(live_db, "DB_PATH", db)
    monkeypatch.setattr(lrj, "DB_PATH", db)
    conn = live_db.connect()
    live_db.upsert_rows(conn, [{"date": "2026-09-16", "race_id": "rac_card", "horse": "Abundant"}])
    conn.close()
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"results": [dict(RACE, race_id="rac_3", date="2026-09-16")]}))
    monkeypatch.setattr("sys.argv", ["x", str(f)])
    assert lrj.main() == 0
    row = sqlite3.connect(db).execute("select horse, pos from data where date='2026-09-16'").fetchall()
    assert row == [("Abundant (IRE)", 1)]   # matched on the bare name and renamed
