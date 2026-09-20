"""fast_inference: entity counts, walk-forward history and pre-race blanking (tiny temp databases)."""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fast_inference as fi  # noqa: E402
import inference as inf  # noqa: E402

COLS = ["date", "course", "race_id", "off", "race_name", "type", "class", "pattern", "age_band", "sex_rest",
        "dist", "going", "ran", "num", "pos", "draw", "horse", "age", "sex", "wgt", "hg", "jockey", "trainer",
        "[or]", "rpr", "ts", "sire", "dam", "damsire"]


def make_db(path, rows):
    con = sqlite3.connect(path)
    con.execute(f"CREATE TABLE data ({', '.join(COLS)})")
    for r in rows:
        vals = {"course": "Ascot", "race_id": 1, "off": "14:00", "race_name": "x", "type": "Flat", "class": "Class 4",
                "pattern": "", "age_band": "3yo+", "sex_rest": "", "dist": "1m", "going": "Good", "ran": 8, "num": 1,
                "draw": 1, "age": 4, "sex": "G", "wgt": "9-0", "hg": "", "trainer": "T", "or": 80, "sire": "S",
                "dam": "D", "damsire": "DS", "horse": "H", **r}
        con.execute(f"INSERT INTO data VALUES ({','.join('?' for _ in COLS)})",
                    [vals.get(c.strip('[]'), None) for c in COLS])
    con.commit(); con.close()


@pytest.fixture
def dbs(tmp_path, monkeypatch):
    hist, live = tmp_path / "hist.db", tmp_path / "live.db"
    make_db(hist, [
        {"date": "2020-06-01", "jockey": "J", "pos": 1, "race_id": 1},           # before HISTORY_START: ignored
        {"date": "2024-06-01", "jockey": "J", "pos": 1, "race_id": 2},
        {"date": "2024-06-02", "jockey": "J", "pos": "3", "race_id": 3},
        {"date": "2024-06-03", "jockey": "J", "pos": 2, "race_id": 4, "ran": 2},  # ran < 3: ignored
        {"date": "2024-06-04", "jockey": "J", "pos": 1, "race_id": 5, "course": "Longchamp (FR)"},  # course filter
        {"date": "2024-06-05", "jockey": None, "pos": 1, "race_id": 6},
    ])
    make_db(live, [
        {"date": "2026-06-01", "jockey": "J", "pos": 1, "race_id": "rac_1"},     # gap day, before the target
        {"date": "2026-06-02", "jockey": "J", "pos": 1, "race_id": "rac_2"},     # the target day itself: excluded
        {"date": "2026-06-03", "jockey": "J", "pos": 1, "race_id": "rac_3"},     # after the target: excluded
    ])
    monkeypatch.setattr(inf, "HISTORICAL_DB", hist)
    monkeypatch.setattr(inf, "LIVE_DB", live)
    return hist, live


def test_entity_counts_are_strictly_before_the_date_from_both_databases(dbs):
    today = pd.DataFrame([{"jockey": "J", "trainer": "T", "course": "Ascot", "type": "Flat",
                           "sire": "S", "dam": "D", "damsire": "DS"}])
    c = fi.entity_counts(today, "2026-06-02")
    jky = c["jky"].set_index("jockey")
    # 2024-06-01 (win), 2024-06-02 (3rd), gap day 2026-06-01 (win); not 2020, not ran<3, not the FR course
    assert jky.loc["J", "jky_runs"] == 3 and jky.loc["J", "jky_wins"] == 2
    assert c["jt"].iloc[0]["jt_runs"] == 3          # (jockey, trainer) pair
    assert c["tc"].iloc[0]["tc_runs"] == 4          # trainer T at Ascot also counts the null-jockey row


def test_null_key_group_is_counted_like_training(dbs):
    today = pd.DataFrame([{"jockey": None, "trainer": "T", "course": "Ascot", "type": "Flat",
                           "sire": "S", "dam": "D", "damsire": "DS"}])
    c = fi.entity_counts(today, "2026-06-02")
    assert c["jky"]["jky_runs"].iloc[0] == 1        # the single null-jockey row


def test_prerace_view_blanks_results():
    r = pd.DataFrame([{"pos": 1, "rpr": 90, "ts": 80, "horse": "A"}])
    v = fi.prerace_view(r)
    assert v[["pos", "rpr", "ts"]].isna().all().all() and v["horse"].iloc[0] == "A"
    assert r["pos"].iloc[0] == 1                    # the input is not modified


def test_features_for_day_walks_forward_and_new_entities_get_zero(dbs):
    hist, live = dbs
    today = pd.DataFrame([{"date": "2026-06-02", "course": "Ascot", "race_id": "rac_2", "off": "14:00", "race_name": "x",
                           "type": "Flat", "class": "Class 4", "pattern": "", "age_band": "3yo+", "sex_rest": "",
                           "dist": "1m", "going": "Good", "ran": 3, "num": n, "pos": 1, "draw": n, "horse": h, "age": 4,
                           "sex": "G", "wgt": "9-0", "hg": "", "jockey": j, "trainer": "T2", "or": 80, "rpr": 5, "ts": 5,
                           "sire": "S2", "dam": "D2", "damsire": "DS2"}
                          for n, h, j in [(1, "A (GB)", "J"), (2, "B (GB)", "NEWJOCKEY"), (3, "C (GB)", "J")]])
    f = fi.features_for_day("2026-06-02", today)
    assert len(f) == 3
    by = f.set_index("horse")
    assert by.loc["A (GB)", "jky_runs"] == 3                  # J: two historic rows + the gap day before the target
    assert by.loc["B (GB)", "jky_runs"] == 0 and by.loc["B (GB)", "jky_wins"] == 0   # unseen jockey: 0, not NaN
    assert by["h_runs_prior"].eq(0).all()                     # these horses have no earlier runs
