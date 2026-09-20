"""
Fast per-day inference: features for ONLY the runners of one race day, no whole-history rebuild.

Same model, same features.build() code, same 135 features as inference.py, but instead of
loading ~1M history rows and rebuilding every entity's timeline (minutes, >4 GB) it loads:
  * each running horse's OWN past runs (a few thousand rows; all horse-level features come from
    these: prior form, days since run, EMAs, horse x course/distance/going/type experience), and
  * for every jockey/trainer/sire/dam/damsire/pair key, the running run/win counts strictly
    before the target date (one SQL GROUP BY each; these are exactly what features.build's
    day_stats computes).
Then features.build(..., entity_counts=...) runs on that small frame.

History = raceform.db (to 2026-05-27) UNION live_extension.db rows dated before the target
date, so finished days are part of the history for all later days (walk-forward): predict day D,
score it, and D's results are available to D+1 automatically. History starts at HISTORY_START
(2021-01-01), the window the training features were built with (features.py's default); the old
inference.py loaded everything back to 2015, which inflated every entity count vs training.

Limits: rows loaded from live_extension have rpr/ts NULL (the vendor's ratings are a different
scale; see data_collection/results_mapping.py), so prior_rpr/prior_ts do not see those runs.

    python fast_inference.py --date 2026-09-19 [--out ...]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as feat  # noqa: E402
import inference as inf  # noqa: E402  (blend_predictions, RAW_COLS, model paths)

HISTORY_START = "2021-01-01"   # what features.py used to build the training matrix
COURSE_FILTER = "(course NOT LIKE '%(%' OR course LIKE '%(IRE)' OR course LIKE '%(AW)')"
KEY_SETS = [(["jockey"], "jky"), (["trainer"], "trn"), (["jockey", "trainer"], "jt"),
            (["trainer", "course"], "tc"), (["jockey", "course"], "jc"), (["sire"], "sire"),
            (["dam"], "dam"), (["damsire"], "dsire"), (["trainer", "type"], "tt")]
POST_RACE_COLS = ["pos", "rpr", "ts"]   # the only result columns in RAW_COLS
BASE_WHERE = ("date != 'date' AND date >= ? AND date < ? AND " + COURSE_FILTER +
              " AND CAST(ran AS REAL) >= 3")


def _dbs() -> list[Path]:
    return [inf.HISTORICAL_DB, inf.LIVE_DB]


def _in_clause(col: str, values: list) -> tuple[str, list]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    has_null = len(vals) != len(values)
    parts, params = [], []
    if vals:
        parts.append(f"{col} IN ({','.join('?' for _ in vals)})")
        params += vals
    if has_null:
        parts.append(f"{col} IS NULL")
    return "(" + " OR ".join(parts) + ")" if parts else "0", params


def load_horse_history(horses: list[str], before: str) -> pd.DataFrame:
    frames = []
    for db in _dbs():
        if not db.exists():
            continue
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        cond, params = _in_clause("horse", horses)
        frames.append(pd.read_sql(
            f"SELECT {inf.RAW_COLS} FROM data WHERE {BASE_WHERE} AND {cond}", con,
            params=[HISTORY_START, before, *params]))
        con.close()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["date", "race_id", "horse"]).reset_index(drop=True)


def entity_counts(today: pd.DataFrame, before: str) -> dict[str, pd.DataFrame]:
    """{tag: DataFrame(keys, tag_runs, tag_wins)}: rows/wins strictly before `before`."""
    out = {}
    for keys, tag in KEY_SETS:
        conds, params = [], []
        for k in keys:
            c, p = _in_clause(k, today[k].unique().tolist())
            conds.append(c); params += p
        cols = ", ".join(f"[{k}]" for k in keys)
        sql = (f"SELECT {cols}, COUNT(*) AS n, "
               f"SUM(CASE WHEN TRIM(CAST(pos AS TEXT)) = '1' THEN 1 ELSE 0 END) AS w "
               f"FROM data WHERE {BASE_WHERE} AND {' AND '.join(conds)} GROUP BY {cols}")
        frames = []
        for db in _dbs():
            if not db.exists():
                continue
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            frames.append(pd.read_sql(sql, con, params=[HISTORY_START, before, *params]))
            con.close()
        df = pd.concat(frames, ignore_index=True)
        df[["n", "w"]] = df[["n", "w"]].astype("float64")
        df = df.groupby(keys, dropna=False, as_index=False)[["n", "w"]].sum()
        out[tag] = df.rename(columns={"n": f"{tag}_runs", "w": f"{tag}_wins"})
    return out


def prerace_view(rows: pd.DataFrame) -> pd.DataFrame:
    """Blank the result columns so nothing from the outcome can reach the features."""
    rows = rows.copy()
    for c in POST_RACE_COLS:
        if c in rows.columns:
            rows[c] = np.nan
    return rows


def features_for_day(date: str, today_rows: pd.DataFrame) -> pd.DataFrame:
    today_rows = prerace_view(today_rows)
    hist = load_horse_history(today_rows["horse"].dropna().unique().tolist(), date)
    counts = entity_counts(today_rows, date)
    combined = pd.concat([hist, today_rows], ignore_index=True)
    built = feat.build(combined, drop_zero_winner_races=False, entity_counts=counts)
    return built[built["date"] == pd.Timestamp(date)].reset_index(drop=True)


def predict_day(date: str, today_rows: pd.DataFrame) -> pd.DataFrame:
    today = features_for_day(date, today_rows)
    if today.empty:
        raise RuntimeError(f"no rows survived features.build() for {date}")
    feats = [l.strip() for l in open(HERE / "cache" / "feature_names.txt") if l.strip()]
    feats = [f for f in feats if f in today.columns]
    return inf.blend_predictions(today, feats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    live = inf.load_live(args.date)
    if live.empty:
        print(f"No rows for {args.date} in {inf.LIVE_DB}", file=sys.stderr)
        return 1
    preds = predict_day(args.date, live)
    out = Path(args.out) if args.out else HERE / "predictions" / f"predictions_{args.date}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out, index=False)
    print(f"Wrote {len(preds)} predictions ({preds['race_id'].nunique()} races) to {out}\n")
    for _, g in preds.groupby("race_id", sort=False):
        head = g.sort_values("blend_all", ascending=False).iloc[0]
        print(f"{head['course']:>14} {head['off']:>6}  {head['race_name'][:45]:<45}  "
              f"-> {head['horse']} (p={head['blend_all']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
