"""
Score the walk-forward predictions (predictions/backtest_wf/predictions_<date>.csv, written by
backtest_walkforward.py) against the settled results now
in data/live_extension.db. Prints a markdown report and writes report.md + threshold CSVs.

    python evaluate_backtest.py [--dir predictions/backtest_wf] [--boot 1000]
"""
from __future__ import annotations

import argparse
import glob
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
LIVE_DB = HERE.parent.parent / "data" / "live_extension.db"
# held-out test window 2025-11-29..2026-05-27, blend_all (docs/PROJECT_NOTES.md sections 3, 6.4)
REF = {"top-1": 0.273, "AUC": 0.7445}
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def sp_to_decimal(sp) -> float:
    """'11/4F' -> 3.75, 'evens'/'evs' -> 2.0, anything else -> NaN (stake included)."""
    if sp is None or (isinstance(sp, float) and np.isnan(sp)):
        return np.nan
    s = re.sub(r"[A-Za-z]+$", "", str(sp).strip()) if re.search(r"\d", str(sp)) else str(sp).strip().lower()
    if s in ("evens", "evs"):
        return 2.0
    m = re.match(r"^(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$", s)
    return float(m.group(1)) / float(m.group(2)) + 1.0 if m else np.nan


def load(pred_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(pred_dir / "predictions_*.csv")))
    preds = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    con = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
    dates = tuple(sorted(preds["date"].unique()))
    res = pd.read_sql(
        f"SELECT date, race_id, horse, pos, sp, type, ran, dist FROM data WHERE date IN ({','.join('?' * len(dates))})",
        con, params=dates)
    con.close()
    df = preds.merge(res, on=["date", "race_id", "horse"], how="inner")
    df["finish"] = pd.to_numeric(df["pos"], errors="coerce")
    df["win"] = (df["finish"] == 1).astype(int)
    df["placed"] = (df["finish"] <= 3).astype(int)
    df["sp_dec"] = df["sp"].map(sp_to_decimal)
    df["region"] = np.where(df["course"].str.contains(r"\(IRE\)"), "IRE", "GB")
    df["month"] = df["date"].str[:7]
    # a race whose winner is absent from the results (e.g. dead heat / void) is not scoreable
    ok = df.groupby("race_id")["win"].transform("sum") >= 1
    return df[ok].reset_index(drop=True)


def race_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["race_id", "rank_in_race"])
    top = d.groupby("race_id").head(1).set_index("race_id")
    top3 = d.groupby("race_id").head(3).groupby("race_id")["placed"].sum()
    field = d.groupby("race_id").size()
    win_rank = d[d["win"] == 1].groupby("race_id")["rank_in_race"].min()
    out = pd.DataFrame({
        "date": top["date"], "month": top["month"], "region": top["region"], "type": top["type"],
        "field": field, "top1": top["win"], "top3": top["placed"], "p1": top["blend_all"],
        "sp1": top["sp_dec"], "hits3": top3, "n3": d.groupby("race_id").head(3).groupby("race_id").size(),
        "win_rank": win_rank,
    })
    return out


def headline(rt: pd.DataFrame, df: pd.DataFrame) -> dict:
    fin = df[df["finish"].notna()]
    return {
        "races": len(rt), "runners": len(df), "top1": rt["top1"].mean(), "top3": rt["top3"].mean(),
        "p_at_3": rt["hits3"].sum() / rt["n3"].sum(), "mrr": (1.0 / rt["win_rank"]).mean(),
        "mean_pos_err": (fin["rank_in_race"] - fin["finish"]).abs().mean(),
        "auc": roc_auc_score(df["win"], df["blend_all"]), "ap": average_precision_score(df["win"], df["blend_all"]),
        "logloss": log_loss(df["win"], df["blend_all"].clip(1e-6, 1 - 1e-6)),
        "base_rate": df["win"].mean(),
    }


def bootstrap(rt: pd.DataFrame, df: pd.DataFrame, n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ids = rt.index.to_numpy()
    by_race = {rid: g for rid, g in df.groupby("race_id")}
    vals = {"top1": [], "top3": [], "p_at_3": [], "ap": []}
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        r = rt.loc[pick]
        vals["top1"].append(r["top1"].mean()); vals["top3"].append(r["top3"].mean())
        vals["p_at_3"].append(r["hits3"].sum() / r["n3"].sum())
        # AP on a resample is the expensive one; use runner rows of the resampled races
        sub = pd.concat([by_race[i] for i in pick[: min(len(pick), 1500)]])
        vals["ap"].append(average_precision_score(sub["win"], sub["blend_all"]))
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in vals.items()}


def runner_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    rows, pos = [], df["win"].sum()
    for t in THRESHOLDS:
        flag = df["blend_all"] >= t
        tp = int((flag & (df["win"] == 1)).sum()); n = int(flag.sum())
        prec = tp / n if n else np.nan; rec = tp / pos
        f1 = 2 * prec * rec / (prec + rec) if n and tp else 0.0
        f = df[flag & df["sp_dec"].notna()]
        roi = (f["win"] * f["sp_dec"]).sum() / len(f) - 1 if len(f) else np.nan
        rows.append({"threshold": t, "flagged": n, "per_race": n / df["race_id"].nunique(),
                     "precision": prec, "recall": rec, "F1": f1, "ROI_at_SP": roi})
    return pd.DataFrame(rows)


def toppick_thresholds(rt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in [0.0] + THRESHOLDS:
        r = rt[rt["p1"] >= t]
        if r.empty:
            continue
        s = r[r["sp1"].notna()]
        roi = (s["top1"] * s["sp1"]).sum() / len(s) - 1 if len(s) else np.nan
        rows.append({"min_p_of_pick": t, "races": len(r), "coverage": len(r) / len(rt),
                     "top1": r["top1"].mean(), "top3": r["top3"].mean(), "ROI_at_SP": roi})
    return pd.DataFrame(rows)


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy(); d["bin"] = pd.qcut(d["blend_all"], 10, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(n=("win", "size"), mean_p=("blend_all", "mean"), win_rate=("win", "mean"))
    return g.reset_index(drop=True)


def group_rows(rt: pd.DataFrame, col: str) -> pd.DataFrame:
    g = rt.groupby(col).agg(races=("top1", "size"), top1=("top1", "mean"), top3=("top3", "mean"),
                            p_at_3=("hits3", "sum"), n3=("n3", "sum"))
    g["p_at_3"] = g["p_at_3"] / g.pop("n3")
    return g


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    d = df if isinstance(df.index, pd.RangeIndex) else df.reset_index()
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "predictions" / "backtest_wf"))
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    out_dir = Path(args.dir)

    df = load(out_dir)
    rt = race_table(df)
    h = headline(rt, df)
    ci = bootstrap(rt, df, args.boot)
    days = df["date"].nunique()

    daily = rt.groupby("date").agg(races=("top1", "size"), top1=("top1", "mean"), top3=("top3", "mean"),
                                   hits3=("hits3", "sum"), n3=("n3", "sum"))
    daily["p_at_3"] = daily["hits3"] / daily["n3"]
    pct = daily[["top1", "top3", "p_at_3"]].quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).T
    pct.columns = [f"p{int(q * 100)}" for q in pct.columns]
    pct.insert(0, "mean", daily[["top1", "top3", "p_at_3"]].mean())
    pct.insert(1, "sd", daily[["top1", "top3", "p_at_3"]].std())
    rt["field_bucket"] = pd.cut(rt["field"], [0, 8, 12, 16, 100], labels=["<=8", "9-12", "13-16", "17+"])

    lines = [
        f"# Backtest of live inference on the post-May gap ({df['date'].min()} to {df['date'].max()})",
        f"{days} days, {h['races']} GB+IRE races, {h['runners']} runners. Walk-forward with fast_inference.py: each day is "
        "predicted from history = raceform.db (to 2026-05-27) plus every earlier day's results, with the day's own result columns "
        "blanked. Features verified identical to the training matrix. `rpr`/`ts` are NULL for the gap days (prior_rpr/prior_ts "
        "do not see them).\n",
        "## Headline (#1 pick per race, blend_all)",
        "| metric | value | 95% CI (race bootstrap) | held-out reference |", "|---|---|---|---|",
        f"| top-1 (pick won) | {h['top1']:.3f} | {ci['top1'][0]:.3f} to {ci['top1'][1]:.3f} | {REF['top-1']} |",
        f"| top-3 (pick placed) | {h['top3']:.3f} | {ci['top3'][0]:.3f} to {ci['top3'][1]:.3f} | ~0.60 |",
        f"| precision@3 | {h['p_at_3']:.3f} | {ci['p_at_3'][0]:.3f} to {ci['p_at_3'][1]:.3f} | |",
        f"| MRR of the winner | {h['mrr']:.3f} | | ~0.49 |",
        f"| mean position error | {h['mean_pos_err']:.3f} | | random ~3.5 |",
        f"| runner AUC (win) | {h['auc']:.4f} | | {REF['AUC']} |",
        f"| average precision (win) | {h['ap']:.3f} | {ci['ap'][0]:.3f} to {ci['ap'][1]:.3f} | ~0.27 |",
        f"| base win rate per runner | {h['base_rate']:.3f} | | |",
        f"| race log-loss (runner, blend_all) | {h['logloss']:.3f} | | |\n",
        "## Runner-level precision / recall by threshold on blend_all (flag runner if p >= threshold; positive = wins)",
        md(runner_thresholds(df)) + "\n",
        "## #1 pick only, by minimum confidence",
        md(toppick_thresholds(rt)) + "\n",
        "## Calibration (deciles of blend_all)", md(calibration(df)) + "\n",
        "## By month", md(group_rows(rt, "month")) + "\n",
        "## By region", md(group_rows(rt, "region")) + "\n",
        "## By field size", md(group_rows(rt, "field_bucket")) + "\n",
        "## By race type", md(group_rows(rt, "type")) + "\n",
        f"## Daily distribution (n={len(daily)} days)",
        f"top-1: mean {daily['top1'].mean():.3f}, sd {daily['top1'].std():.3f}, min {daily['top1'].min():.3f}, "
        f"median {daily['top1'].median():.3f}, max {daily['top1'].max():.3f}; "
        f"top-3: mean {daily['top3'].mean():.3f}, sd {daily['top3'].std():.3f}\n",
        "Percentiles of the per-day metric:", md(pct) + "\n",
    ]
    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report)
    runner_thresholds(df).to_csv(out_dir / "thresholds_runner.csv", index=False)
    toppick_thresholds(rt).to_csv(out_dir / "thresholds_toppick.csv", index=False)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
