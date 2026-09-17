"""
Score a day's predictions (from inference.py) against actual results, once
the races have run.

⚠️ AVAILABILITY: this only works for TODAY's date. The Free Racing API plan
exposes finished results only through /v1/results/today/free — there is no
way to fetch a PAST day's results without a Standard plan subscription (the
historical /v1/results endpoint 401s "Standard Plan required", confirmed
live 2026-09-16; see docs/2026-09-14_live_data_pipeline_plan.md). Run this
after the day's last race has gone off, for the same --date you ran
inference.py with earlier that day — waiting until tomorrow makes that
day's results permanently unscoreable on this plan.

Metrics reported, with explicit semantics (nothing here is a standard,
universally-agreed name, so definitions are spelled out):

  * top1_accuracy: fraction of races where the model's #1-ranked pick
    (rank_in_race == 1 in the predictions CSV) actually finished 1st.
    This is "did the favourite-by-model win", not "did the model predict
    every horse's exact position".
  * top3_accuracy: fraction of races where the model's #1-ranked pick
    finished in the top 3 (used each-way in UK/IRE racing).
  * mean_position_error: mean(|predicted rank_in_race - actual finishing
    position|) across every FINISHED runner (non-finishers - fell, pulled
    up, unseated, etc. - have no numeric finishing position and are
    excluded from this average, but are still counted in top1/top3 checks
    as "did not win"). Lower is better; 0 would mean the model's ranking
    exactly matched the finishing order every time.

Usage:
    python score_predictions.py                       # today, once races are done
    python score_predictions.py --date 2026-09-17 \
        --predictions predictions/predictions_2026-09-17.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import date as date_cls
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "data_collection"))
from racingapi_client import RacingAPIClient, RacingAPIError  # noqa: E402


def fetch_results_today() -> pd.DataFrame:
    """
    Flatten racingapi_client.results_today_free() into one row per finished
    runner: race_id, num (matches predictions' 'num' column), position
    (numeric, NaN if not a finisher).
    """
    client = RacingAPIClient()
    races = client.results_today_free()
    rows = []
    for race in races:
        for runner in race.get("runners", []):
            pos_raw = runner.get("position")
            try:
                pos = float(pos_raw)
            except (TypeError, ValueError):
                pos = np.nan  # non-finisher: PU, F, UR, RO, BD, ...
            rows.append({
                "race_id": race.get("race_id"),
                "num": runner.get("number"),
                "horse": runner.get("horse"),
                "position": pos,
                "position_raw": pos_raw,
            })
    return pd.DataFrame(rows)


def score(predictions: pd.DataFrame, results: pd.DataFrame) -> dict:
    predictions = predictions.copy()
    predictions["num"] = predictions["num"].astype(str)
    results = results.copy()
    results["num"] = results["num"].astype(str)

    merged = predictions.merge(results, on=["race_id", "num"], how="inner", suffixes=("", "_result"))
    n_races_predicted = predictions["race_id"].nunique()
    n_races_matched = merged["race_id"].nunique()
    if n_races_matched == 0:
        raise RuntimeError(
            "No races matched between predictions and results — either none of "
            "today's races have finished yet, or the (race_id, num) join key "
            "doesn't line up (check inference.py's output still includes 'num')."
        )

    top_picks = merged[merged["rank_in_race"] == 1]
    top1_accuracy = float((top_picks["position"] == 1).mean())
    top3_accuracy = float((top_picks["position"] <= 3).mean())

    finished = merged[merged["position"].notna()]
    mean_position_error = float((finished["rank_in_race"] - finished["position"]).abs().mean())

    return {
        "date": predictions["date"].iloc[0],
        "n_races_predicted": n_races_predicted,
        "n_races_matched": n_races_matched,
        "n_races_unfinished_or_unmatched": n_races_predicted - n_races_matched,
        "n_runners_matched": len(merged),
        "n_runners_finished": len(finished),
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "mean_position_error": mean_position_error,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date_cls.today().isoformat())
    ap.add_argument("--predictions", default=None,
                     help="default: predictions/predictions_<date>.csv")
    args = ap.parse_args()

    pred_path = Path(args.predictions) if args.predictions else HERE / "predictions" / f"predictions_{args.date}.csv"
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found — run inference.py --date {args.date} first.",
              file=sys.stderr)
        return 1
    predictions = pd.read_csv(pred_path)

    if args.date != date_cls.today().isoformat():
        print(f"WARNING: --date {args.date} is not today ({date_cls.today().isoformat()}). "
              f"/v1/results/today/free only ever returns TODAY's results on the Free plan "
              f"— this will find zero matches unless {args.date} happens to equal today's "
              f"server date. See this script's module docstring.", file=sys.stderr)

    try:
        results = fetch_results_today()
    except RacingAPIError as e:
        print(f"ERROR fetching results: {e}", file=sys.stderr)
        return 1
    if results.empty:
        print("No results returned yet — races may not have finished. Try again later.",
              file=sys.stderr)
        return 1

    try:
        m = score(predictions, results)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Scored {m['n_races_matched']}/{m['n_races_predicted']} races "
          f"({m['n_races_unfinished_or_unmatched']} not yet finished / unmatched), "
          f"{m['n_runners_finished']}/{m['n_runners_matched']} runners with a known finish position.\n")
    print(f"top1_accuracy        = {m['top1_accuracy']:.3f}  "
          f"(model's #1 pick actually won this fraction of races)")
    print(f"top3_accuracy        = {m['top3_accuracy']:.3f}  "
          f"(model's #1 pick finished in the top 3 this fraction of races)")
    print(f"mean_position_error  = {m['mean_position_error']:.3f}  "
          f"(mean |predicted rank - actual finishing position| across finishers; lower is better, 0 = perfect)")

    out_path = HERE / "predictions" / f"score_{m['date']}.csv"
    pd.DataFrame([m]).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
