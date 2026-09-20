"""check_selections.settle(): each-way Lucky 15 settlement from a /v1/results-shaped payload."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import check_selections as cs  # noqa: E402

SLIP = json.load(open(Path(__file__).resolve().parent.parent / "slips" / "2026-09-20_lucky15.json"))


def race(course, off, runners):
    return {"course": course, "off_dt": f"2026-09-20T{off}:00+01:00",
            "runners": [{"horse": h, "position": pos, "sp_dec": sp} for h, pos, sp in runners]}


def results(positions):
    """positions: list of finishing positions for the four legs, None = race not run."""
    out = []
    for leg, pos in zip(SLIP["legs"], positions):
        if pos is None:
            continue
        out.append(race(leg["course"] + " (IRE)", leg["off"], [(leg["horse"].split(" (")[0], str(pos), "3.0"), ("Other", "9", "9.0")]))
    return out


def test_all_four_win_matches_the_hand_calculation():
    s = cs.settle(SLIP, results([1, 1, 1, 1]))
    assert s["done"] and s["outcomes"] == ["W"] * 4
    assert s["returns"] == pytest.approx(0.16 * 543.8, rel=1e-3)      # 30 lines, 543.8 units at 1 per line
    assert s["stake"] == pytest.approx(4.8)


def test_place_terms_and_losers():
    # Le Nez Creux 2nd (2 places -> placed), Premier Fantasy 3rd (2 places -> lost), Haveanothertry 3rd (3 places -> placed), Shane's 1st
    s = cs.settle(SLIP, results([2, 3, 3, 1]))
    assert s["outcomes"] == ["P", "L", "P", "W"]


def test_nothing_places_loses_the_whole_stake():
    s = cs.settle(SLIP, results([5, 6, 7, 8]))
    assert s["returns"] == 0 and s["net"] == pytest.approx(-4.8)


def test_pending_race_is_not_done():
    s = cs.settle(SLIP, results([1, 1, 1, None]))
    assert not s["done"] and s["outcomes"][3] == "pending"


def test_withdrawn_horse_makes_the_leg_void():
    res = results([1, 1, 1, 1])
    res[0]["runners"] = [{"horse": "Someone Else", "position": "1", "sp_dec": "2.0"}]   # our horse is not in the results
    s = cs.settle(SLIP, res)
    assert s["outcomes"][0] == "NR"
