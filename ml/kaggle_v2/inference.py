"""
Live inference: score today's (or any date's) racecard with the trained
blend_all model (binary + race-softmax + Plackett-Luce top-3, UK+IRE
combined — the "base", no-TAG model in cache/, matching results_final.csv's
blend_all row, AUC 0.7445 on the held-out test split; see
docs/PROJECT_NOTES.md, section 3).

Pipeline:
  1. Load TODAY's pre-race rows from data/live_extension.db (written by
     data_collection/fetch_daily_racecards.py) - pos/rpr/ts/sp/prize are
     NULL there, which is correct: they don't exist yet.
  2. Load ONLY the historical rows that can affect today's field: every row
     whose horse/jockey/trainer/sire/dam/damsire matches an entity running
     today (data_ext/raceform.db has indexes on all six columns for this -
     see docs/PROJECT_NOTES.md, section 5). Every rolling/entity
     stat in features.py (day_stats, a horse's own prior_* shifts,
     horse-x-course/dist/going/type experience) is computed per-entity, so
     an entity's cumulative stat depends only on ITS OWN historical rows -
     rows for entities not running today cannot affect today's features and
     are correctly never loaded. This is what makes inference cheap: ~180
     horses/day need their own history, not a full-table scan of the ~1.85M
     row DB.
  3. Concatenate and run features.build() on the combined frame. This is
     safe/correct because build()'s point-in-time logic (day_stats, prior_*
     shifts) naturally treats a NaN pos as "not yet run" and the day-boundary
     cumulative stats already exclude same-day rows by construction - so
     today's rows get properly-computed pre-race features without any
     special-casing.
  4. Filter to today's date, select the 135 base features (cache/feature_names.txt,
     includes is_ire so both GB and IRE races score off one model), run the
     three saved boosters, and blend exactly as exp_final.py's blend() does:
     geometric mean of per-race-normalised probabilities, renormalised.

⚠️ KNOWN DATA GAP (see docs/PROJECT_NOTES.md, section 2.1): the
historical DB ends 2026-05-27 and backfill_history.py is blocked on a
Standard Racing API plan, so entity/rolling stats (days_since_run,
jky_runs/trn_wr/etc., a horse's own recent-form features) do not reflect
anything that happened between 2026-05-28 and yesterday. Predictions are
directionally usable but should be treated as running on stale recent-form
data until the backfill gap is closed. This script prints that caveat and
the size of the gap every run so it can't be silently forgotten.

Usage:
    python inference.py                  # today
    python inference.py --date 2026-09-17
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date as date_cls
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent          # ml/kaggle_v2
REPO_ROOT = HERE.parent.parent                    # repo root
sys.path.insert(0, str(HERE))
import features as feat  # noqa: E402
from common import softmax_by_group, make_group_index  # noqa: E402

RAW_COLS = ("date, course, race_id, off, race_name, type, class, pattern, age_band, "
            "sex_rest, dist, going, ran, num, pos, draw, horse, age, sex, wgt, hg, "
            "jockey, trainer, [or], rpr, ts, sire, dam, damsire")

HISTORICAL_DB = REPO_ROOT / "data_ext" / "raceform.db"
LIVE_DB = REPO_ROOT / "data" / "live_extension.db"

MODELS = ["binary", "softmax", "pl_top3"]
MODEL_FILES = {
    "binary": HERE / "cache" / "final_binary.json",
    "softmax": HERE / "cache" / "final_softmax.json",
    "pl_top3": HERE / "cache" / "final_pltop3.json",
}
MODEL_KIND = {"binary": "prob", "softmax": "score", "pl_top3": "score"}


ENTITY_COLS = ["horse", "jockey", "trainer", "sire", "dam", "damsire"]
RAW_COL_LIST = [c.strip().strip("[]") for c in RAW_COLS.split(",")]


def load_historical_for_entities(entities: dict[str, list[str]], before_date: str) -> pd.DataFrame:
    """
    Load only historical rows that can affect an entity running today: for
    each of horse/jockey/trainer/sire/dam/damsire, every past row where that
    column matches one of today's values. One SQL query per entity column
    (kept separate rather than one big OR, to stay well under SQLite's
    default 999-bound-parameter limit for a day with many runners), unioned
    and de-duplicated on (date, race_id, horse) since a historical row can
    legitimately match more than one entity column (e.g. same jockey AND
    trainer as today) and would otherwise be double-counted.
    """
    con = sqlite3.connect(HISTORICAL_DB)
    frames = []
    for col in ENTITY_COLS:
        values = sorted({v for v in entities.get(col, []) if v})
        if not values:
            continue
        placeholders = ",".join("?" for _ in values)
        q = (f"SELECT {RAW_COLS} FROM data WHERE date != 'date' AND date < ? "
             f"AND (course NOT LIKE '%(%' OR course LIKE '%(IRE)' OR course LIKE '%(AW)') "
             f"AND {col} IN ({placeholders})")
        frames.append(pd.read_sql(q, con, params=[before_date, *values]))
    con.close()
    if not frames:
        return pd.DataFrame(columns=RAW_COL_LIST)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "race_id", "horse"]).reset_index(drop=True)
    return combined


def load_live(target_date: str) -> pd.DataFrame:
    if not LIVE_DB.exists():
        raise FileNotFoundError(
            f"{LIVE_DB} not found — run "
            f"`data_collection/fetch_daily_racecards.py --day today` first."
        )
    con = sqlite3.connect(LIVE_DB)
    q = (f"SELECT {RAW_COLS} FROM data WHERE date = '{target_date}' "
         f"AND (course NOT LIKE '%(%' OR course LIKE '%(IRE)' OR course LIKE '%(AW)')")
    df = pd.read_sql(q, con)
    con.close()
    return df


def report_data_gap(historical: pd.DataFrame, target_date: str) -> None:
    if historical.empty:
        print("⚠️  No historical rows matched any of today's entities — "
              "every runner/jockey/trainer today is new to this DB.", file=sys.stderr)
        return
    hist_max = historical.loc[historical["date"] != "date", "date"].max()
    gap_days = (pd.Timestamp(target_date) - pd.Timestamp(hist_max)).days
    print(f"⚠️  Historical data ends {hist_max}; today is {target_date} "
          f"({gap_days} days of unfilled history — backfill_history.py is blocked "
          f"on a Standard Racing API plan, see docs/PROJECT_NOTES.md, section 2.1).\n"
          f"    Recent-form features (days_since_run, entity day_stats, prior_*) for any "
          f"horse/jockey/trainer that has run since {hist_max} will UNDERSTATE recent "
          f"activity. Treat predictions as directional, not calibrated, until this gap "
          f"is backfilled.\n", file=sys.stderr)


def blend_predictions(today: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    X = today[feats].to_numpy(np.float32)
    dtoday = xgb.DMatrix(X, missing=np.nan)
    rid = today["race_id"].to_numpy()
    gidx = make_group_index(rid)
    n_groups = gidx.max() + 1

    pn = {}
    for name in MODELS:
        booster = xgb.Booster()
        booster.load_model(str(MODEL_FILES[name]))
        raw = booster.predict(dtoday)
        if MODEL_KIND[name] == "score":
            pn[name] = softmax_by_group(raw.astype(np.float64), gidx, n_groups)
        else:  # "prob" (binary) -> normalise within race to sum 1
            s = pd.Series(raw, dtype=np.float64).groupby(rid).transform("sum").to_numpy()
            pn[name] = raw.astype(np.float64) / s

    # geometric mean of the three, renormalised per race (matches exp_final.py's blend())
    log_blend = sum(np.log(np.clip(pn[n], 1e-12, None)) for n in MODELS) / len(MODELS)
    pn["blend_all"] = softmax_by_group(log_blend, gidx, n_groups)

    out = today[["date", "race_id", "course", "off", "race_name", "horse", "jockey", "trainer", "num"]].copy()
    for name in MODELS + ["blend_all"]:
        out[name] = pn[name]
    out["rank_in_race"] = out.groupby("race_id")["blend_all"].rank(ascending=False, method="first")
    return out.sort_values(["date", "course", "off", "rank_in_race"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date_cls.today().isoformat())
    ap.add_argument("--out", default=None, help="CSV path (default: predictions/predictions_<date>.csv)")
    args = ap.parse_args()

    for name, path in MODEL_FILES.items():
        if not path.exists():
            print(f"ERROR: missing model file {path} — run exp_final.py first.", file=sys.stderr)
            return 1

    print(f"Loading today's racecard ({args.date}) from {LIVE_DB} ...", flush=True)
    try:
        live = load_live(args.date)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if live.empty:
        print(f"No racecard rows for {args.date} in {LIVE_DB} — fetch it first.", file=sys.stderr)
        return 1
    print(f"  {len(live)} runner rows across {live['race_id'].nunique()} races.")

    entities = {col: live[col].dropna().unique().tolist() for col in ENTITY_COLS if col in live.columns}
    print(f"Loading history scoped to today's entities from {HISTORICAL_DB} "
          f"({', '.join(f'{len(v)} {k}s' for k, v in entities.items())}) ...", flush=True)
    historical = load_historical_for_entities(entities, before_date=args.date)
    print(f"  {len(historical)} historical rows loaded (vs ~1.85M in the full table).")
    report_data_gap(historical, args.date)

    combined = pd.concat([historical, live], ignore_index=True)
    print("Building point-in-time features (this can take a minute)...", flush=True)
    built = feat.build(combined)

    today_df = built[built["date"] == pd.Timestamp(args.date)].reset_index(drop=True)
    if today_df.empty:
        print(f"ERROR: no rows survived features.build() for {args.date} "
              f"(check ran>=3 / exactly-one-winner filters aren't dropping the live rows "
              f"— today's rows have no winner yet, verify build() doesn't require one for "
              f"unfinished races).", file=sys.stderr)
        return 1

    feats_path = HERE / "cache" / "feature_names.txt"
    feats = [l.strip() for l in open(feats_path) if l.strip()]
    feats = [f for f in feats if f in today_df.columns]
    missing = [f for f in feats if today_df[f].isna().all()]
    if missing:
        print(f"NOTE: {len(missing)} feature(s) are all-NaN for today's rows: {missing[:10]}"
              f"{'...' if len(missing) > 10 else ''}", file=sys.stderr)

    preds = blend_predictions(today_df, feats)

    out_path = Path(args.out) if args.out else HERE / "predictions" / f"predictions_{args.date}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    print(f"\nWrote {len(preds)} predictions to {out_path}\n")

    for race_id, g in preds.groupby("race_id", sort=False):
        head = g.sort_values("blend_all", ascending=False).iloc[0]
        print(f"{head['course']:>14} {head['off']:>6}  {head['race_name'][:45]:<45}  "
              f"-> {head['horse']} (p={head['blend_all']:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
