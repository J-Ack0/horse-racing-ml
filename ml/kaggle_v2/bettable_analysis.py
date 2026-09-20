"""
How much of the model's output is at a bettable price? Uses the walk-forward backtest
(predictions/backtest_wf) joined to the starting prices (closing odds) in live_extension.db.

For a confidence threshold t, the empirical precision of runners with blend_all >= t gives the
break-even decimal odds 1/precision(t). A pick is "bettable" when the price is at or above that.

    python bettable_analysis.py [--margin 0.0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_backtest as ev  # noqa: E402

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def roi(sub: pd.DataFrame) -> float:
    return float((sub["win"] * sub["sp_dec"]).sum() / len(sub) - 1) if len(sub) else float("nan")


def table(df: pd.DataFrame, margin: float, top_only: bool) -> pd.DataFrame:
    d = df[df["sp_dec"].notna()]
    if top_only:
        d = d[d["rank_in_race"] == 1]
    rows = []
    for t in THRESHOLDS:
        s = d[d["blend_all"] >= t]
        if s.empty:
            continue
        prec = s["win"].mean()
        be = (1.0 / prec) * (1 + margin)
        ok = s[s["sp_dec"] >= be]
        no = s[s["sp_dec"] < be]
        rows.append({"t": t, "picks": len(s), "precision": prec, "break_even_odds": be,
                     "bettable_n": len(ok), "bettable_%": len(ok) / len(s),
                     "bettable_win%": ok["win"].mean() if len(ok) else np.nan, "bettable_ROI": roi(ok),
                     "not_bettable_win%": no["win"].mean() if len(no) else np.nan, "not_bettable_ROI": roi(no),
                     "all_ROI": roi(s)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.0, help="require odds >= break-even * (1+margin)")
    args = ap.parse_args()
    df = ev.load(HERE / "predictions" / "backtest_wf")
    print(f"{df['race_id'].nunique()} races, {len(df)} runners, {df['sp_dec'].notna().mean():.1%} with an SP\n")
    fmt = lambda d: ev.md(d, "{:.3f}")
    print("## Every runner with p >= t (bettable = SP >= 1/precision(t))\n" + fmt(table(df, args.margin, False)) + "\n")
    print("## Only the #1 pick per race\n" + fmt(table(df, args.margin, True)) + "\n")

    top = df[(df["rank_in_race"] == 1) & df["sp_dec"].notna()].copy()
    top["fair_odds"] = 1 / top["blend_all"]
    top["edge"] = top["blend_all"] * top["sp_dec"] - 1          # model p x price - 1
    top["sp_band"] = pd.cut(top["sp_dec"], [1, 2, 3, 4, 6, 10, 1000], labels=["<2", "2-3", "3-4", "4-6", "6-10", "10+"])
    g = top.groupby("sp_band", observed=True).agg(picks=("win", "size"), mean_model_p=("blend_all", "mean"),
                                                   implied_by_sp=("sp_dec", lambda x: (1 / x).mean()),
                                                   win_rate=("win", "mean"))
    g["ROI"] = top.groupby("sp_band", observed=True).apply(roi, include_groups=False)
    print("## #1 picks by starting price: does the model's p agree with the market and reality?\n" + fmt(g.reset_index()) + "\n")
    for m in (0.0, 0.05, 0.10, 0.20):
        v = top[top["edge"] >= m]
        print(f"value rule p x SP - 1 >= {m:.2f}: {len(v)} of {len(top)} #1 picks ({len(v) / len(top):.1%}), "
              f"win rate {v['win'].mean():.3f}, ROI {roi(v):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
