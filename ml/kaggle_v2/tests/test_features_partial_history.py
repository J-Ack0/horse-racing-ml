"""build() must keep partial past races when inference asks it to (drop_zero_winner_races=False)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import features as feat  # noqa: E402


def row(date, race_id, horse, pos, ran=10):
    return {"date": date, "course": "Ascot", "race_id": race_id, "off": "14:00", "race_name": "x",
            "type": "Flat", "class": "Class 4", "pattern": "", "age_band": "3yo+", "sex_rest": "",
            "dist": "1m", "going": "Good", "ran": ran, "num": 1, "pos": pos, "draw": 1,
            "horse": horse, "age": 4, "sex": "G", "wgt": "9-0", "hg": "", "jockey": "J", "trainer": "T",
            "or": 80, "rpr": None, "ts": None, "sire": "S", "dam": "D", "damsire": "DS"}


def frame():
    # Past race R1 was loaded only for "Loser" (its winner is not in the frame); today's race has no result yet.
    return pd.DataFrame([
        row("2026-01-01", "R1", "Loser (GB)", 4),
        row("2026-02-01", "R2", "Loser (GB)", 3),
        row("2026-03-01", "R3", None if False else "Loser (GB)", None),
    ])


def test_default_drops_races_without_a_loaded_winner():
    b = feat.build(frame())
    assert set(b["race_id"]) == {"R3"}          # the partial past races are dropped (training behaviour)


def test_inference_mode_keeps_partial_history_and_counts_prior_runs():
    b = feat.build(frame(), drop_zero_winner_races=False)
    today = b[b["race_id"] == "R3"].iloc[0]
    assert set(b["race_id"]) == {"R1", "R2", "R3"}
    assert today["h_runs_prior"] == 2           # both past runs count, winner loaded or not
