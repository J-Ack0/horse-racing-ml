"""
Walk-forward backtest over the days loaded into live_extension.db (2026-05-28 onward).

For each day D in order: predict D's races with fast_inference (history = raceform.db plus every
live_extension.db row dated before D, so all earlier days' results are already in the history),
write predictions/backtest_wf/predictions_<D>.csv. Nothing else is needed to "append" a day: once
D is finished its rows (with results) are simply older than D+1. The target day's own result
columns are blanked before feature building. Resumable (skips days that already have a CSV).

    python backtest_walkforward.py [--start 2026-05-28] [--end 2026-09-19]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fast_inference as fi  # noqa: E402
import inference as inf  # noqa: E402

OUT_DIR = HERE / "predictions" / "backtest_wf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-28")
    ap.add_argument("--end", default="2026-09-19")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{inf.LIVE_DB}?mode=ro", uri=True)
    days = [r[0] for r in con.execute(
        f"SELECT DISTINCT date FROM data WHERE date >= ? AND date <= ? AND {fi.COURSE_FILTER} ORDER BY date",
        (args.start, args.end))]
    con.close()
    print(f"{len(days)} days {days[0]}..{days[-1]}", flush=True)

    failures, t_all = [], time.time()
    for i, d in enumerate(days, 1):
        out = out_dir / f"predictions_{d}.csv"
        if out.exists():
            continue
        t0 = time.time()
        try:
            preds = fi.predict_day(d, inf.load_live(d))
            preds.to_csv(out, index=False)
            status = f"ok {len(preds)} runners / {preds['race_id'].nunique()} races"
        except Exception as e:  # keep going; report at the end
            failures.append(d)
            status = f"FAILED {type(e).__name__}: {e}"
        print(f"[{i}/{len(days)}] {d} {status} {time.time() - t0:.0f}s (elapsed {(time.time() - t_all) / 60:.0f} min)",
              flush=True)
    (out_dir / "failures.txt").write_text("\n".join(failures))
    print(f"done; failures: {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
