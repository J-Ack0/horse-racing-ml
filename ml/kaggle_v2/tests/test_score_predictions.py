"""Unit tests for score_predictions.py's score() — pure function, no live API."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from score_predictions import score  # noqa: E402


def preds(rows):
    return pd.DataFrame(rows)


def results(rows):
    return pd.DataFrame(rows)


def test_perfect_prediction_scores_zero_error_and_full_accuracy():
    p = preds([
        {"date": "2026-09-17", "race_id": "r1", "num": "1", "rank_in_race": 1},
        {"date": "2026-09-17", "race_id": "r1", "num": "2", "rank_in_race": 2},
    ])
    r = results([
        {"race_id": "r1", "num": "1", "position": 1.0},
        {"race_id": "r1", "num": "2", "position": 2.0},
    ])
    m = score(p, r)
    assert m["top1_accuracy"] == 1.0
    assert m["top3_accuracy"] == 1.0
    assert m["mean_position_error"] == 0.0


def test_top_pick_loses_reduces_top1_but_not_necessarily_top3():
    p = preds([
        {"date": "2026-09-17", "race_id": "r1", "num": "1", "rank_in_race": 1},
        {"date": "2026-09-17", "race_id": "r1", "num": "2", "rank_in_race": 2},
    ])
    r = results([
        {"race_id": "r1", "num": "1", "position": 2.0},  # model's #1 pick actually came 2nd
        {"race_id": "r1", "num": "2", "position": 1.0},
    ])
    m = score(p, r)
    assert m["top1_accuracy"] == 0.0
    assert m["top3_accuracy"] == 1.0  # still placed inside top 3
    assert m["mean_position_error"] == 1.0  # both runners off by exactly 1


def test_non_finisher_excluded_from_position_error_but_counts_as_not_won():
    p = preds([
        {"date": "2026-09-17", "race_id": "r1", "num": "1", "rank_in_race": 1},
        {"date": "2026-09-17", "race_id": "r1", "num": "2", "rank_in_race": 2},
    ])
    r = results([
        {"race_id": "r1", "num": "1", "position": None},  # pulled up / fell -> no numeric position
        {"race_id": "r1", "num": "2", "position": 1.0},
    ])
    m = score(p, r)
    assert m["top1_accuracy"] == 0.0  # the non-finisher (model's pick) did not win
    # only runner #2 has a numeric position (rank_in_race=2, actual position=1 -> |2-1|=1)
    assert m["mean_position_error"] == 1.0
    assert m["n_runners_finished"] == 1


def test_unfinished_race_excluded_from_matched_count():
    p = preds([
        {"date": "2026-09-17", "race_id": "r1", "num": "1", "rank_in_race": 1},
        {"date": "2026-09-17", "race_id": "r2", "num": "1", "rank_in_race": 1},
    ])
    r = results([
        {"race_id": "r1", "num": "1", "position": 1.0},
        # r2 not in results yet -> hasn't run
    ])
    m = score(p, r)
    assert m["n_races_predicted"] == 2
    assert m["n_races_matched"] == 1
    assert m["n_races_unfinished_or_unmatched"] == 1


def test_raises_when_nothing_matches():
    p = preds([{"date": "2026-09-17", "race_id": "r1", "num": "1", "rank_in_race": 1}])
    r = results([{"race_id": "r99", "num": "1", "position": 1.0}])
    with pytest.raises(RuntimeError, match="No races matched"):
        score(p, r)


def test_withdrawn_top_pick_is_reranked_not_dropped():
    p = preds([
        {"date": "2026-09-18", "race_id": "r1", "num": 1, "rank_in_race": 1},  # withdrawn
        {"date": "2026-09-18", "race_id": "r1", "num": 2, "rank_in_race": 2},
        {"date": "2026-09-18", "race_id": "r1", "num": 3, "rank_in_race": 3},
    ])
    r = results([
        {"race_id": "r1", "num": "2", "position": 1.0},
        {"race_id": "r1", "num": "3", "position": 2.0},
    ])
    m = score(p, r)
    assert m["n_nonrunners_dropped"] == 1
    assert m["top1_accuracy"] == 1.0          # #2 becomes the effective #1 and won
    assert m["mean_position_error"] == 0.0    # re-ranked order matches finish


def test_float_num_from_predictions_csv_joins_to_string_num_from_api():
    # Real shapes: predictions.csv has num=2.0 (float), API results have num="2".
    p = preds([{"date": "2026-09-18", "race_id": "r1", "num": 2.0, "rank_in_race": 1}])
    r = results([{"race_id": "r1", "num": "2", "position": 1.0}])
    m = score(p, r)
    assert m["n_races_matched"] == 1
    assert m["top1_accuracy"] == 1.0


def test_num_join_key_is_stringified_so_int_vs_str_mismatch_does_not_break_join():
    p = preds([{"date": "2026-09-17", "race_id": "r1", "num": 1, "rank_in_race": 1}])
    r = results([{"race_id": "r1", "num": "1", "position": 1.0}])
    m = score(p, r)
    assert m["n_races_matched"] == 1
