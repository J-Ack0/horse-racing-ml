"""
Backtest the LIVE inference path over every day loaded into live_extension.db
(the 2026-05-28.. gap, after the model's held-out window), one day at a time.

For each date it runs the unmodified inference.py (same features, same models,
same scoped history from raceform.db, so the gap grows exactly as it did live)
against a temporary PRE-RACE copy of that day's rows: pos, btn, ovr_btn, time,
sp, prize, comment, rpr, ts are nulled so nothing from the result can reach the
features. Predictions go to predictions/backtest/predictions_<date>.csv.

Resumable (skips days that already have a CSV), sequential (inference peaks near
2.7 GB and the Pi has no swap), and waits while memory is low or the daily
racingapi-daily-fetch job is running.

    python backtest_gap.py [--start 2026-05-28] [--end 2026-09-19]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LIVE_DB = REPO / "data" / "live_extension.db"
OUT_DIR = HERE / "predictions" / "backtest"  # overridden by --out-dir
NULL_COLS = ["pos", "btn", "ovr_btn", "time", "sp", "prize", "comment", "rpr", "ts"]


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return 0.0


def fetch_job_running() -> bool:
    r = subprocess.run(["systemctl", "--user", "is-active", "racingapi-daily-fetch.service"],
                       capture_output=True, text=True)
    return r.stdout.strip() in ("active", "activating")


def days(start: str, end: str) -> list[str]:
    con = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT DISTINCT date FROM data WHERE date >= ? AND date <= ? "
        "AND (course NOT LIKE '%(%' OR course LIKE '%(IRE)' OR course LIKE '%(AW)') ORDER BY date",
        (start, end)).fetchall()
    con.close()
    return [r[0] for r in rows]


def race_ids(date: str) -> list[str]:
    con = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT DISTINCT race_id FROM data WHERE date = ? "
        "AND (course NOT LIKE '%(%' OR course LIKE '%(IRE)' OR course LIKE '%(AW)') "
        "ORDER BY course, off", (date,)).fetchall()
    con.close()
    return [r[0] for r in rows]


def prerace_db(date: str, dst: Path, rids: list[str] | None = None) -> None:
    con = sqlite3.connect(dst)
    con.execute(f"ATTACH DATABASE '{LIVE_DB}' AS src")
    con.execute("CREATE TABLE data AS SELECT * FROM src.data WHERE date = ?", (date,))
    if rids is not None:
        con.execute("DELETE FROM data WHERE race_id NOT IN (%s)" % ",".join("?" for _ in rids), rids)
    con.commit()
    con.execute("DETACH DATABASE src")
    con.execute("UPDATE data SET " + ", ".join(f"[{c}] = NULL" for c in NULL_COLS))
    con.commit()
    con.close()


CHILD = (
    "import sys; sys.path.insert(0, {here!r}); from pathlib import Path; import inference; "
    "inference.LIVE_DB = Path({db!r}); sys.argv = ['inference.py', '--date', {d!r}, '--out', {out!r}]; "
    "raise SystemExit(inference.main())"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-28")
    ap.add_argument("--end", default="2026-09-19")
    ap.add_argument("--min-mem-gb", type=float, default=4.0)
    ap.add_argument("--chunks", type=int, default=1,
                    help="NOT EXACT above 1: splitting a day's races into separate runs changed blend_all by up to "
                         "0.086 (mean 0.012, 39%% same rank) on 2026-09-06, so some feature uses other races on the "
                         "same day. Keep 1 for real results.")
    ap.add_argument("--every", type=int, default=1, help="only every Nth day")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--strided", action="store_true",
                    help="process days in the order offset 0,2,1,3 of stride 4 so a partial run is spread over the whole period")
    ap.add_argument("--pause-window", default="00:40-02:10",
                    help="don't START a day inside this local-time window (the daily fetch+inference timer runs at 01:00)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    global OUT_DIR
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = days(args.start, args.end)[args.offset::args.every]
    if args.strided:
        todo = [d for off in (0, 2, 1, 3) for d in todo[off::4]]
    w0, w1 = (tuple(map(int, x.split(":"))) for x in args.pause_window.split("-"))
    print(f"{len(todo)} days {todo[0]}..{todo[-1]}", flush=True)
    failures = []
    t_all = time.time()
    for i, d in enumerate(todo, 1):
        out = OUT_DIR / f"predictions_{d}.csv"
        if out.exists():
            continue
        while True:
            now = time.localtime()
            in_window = w0 <= (now.tm_hour, now.tm_min) < w1
            if not (in_window or fetch_job_running() or mem_available_gb() < args.min_mem_gb):
                break
            print(f"  waiting (window {in_window}, mem {mem_available_gb():.1f} GB, fetch job {fetch_job_running()})", flush=True)
            time.sleep(60)
        t0 = time.time()
        rids = race_ids(d)
        parts, ok = [], True
        for k in range(args.chunks):
            chunk = rids[k::args.chunks]
            if not chunk:
                continue
            part = OUT_DIR / f"_part_{d}_{k}.csv"
            with tempfile.TemporaryDirectory() as tmp:
                db = Path(tmp) / "prerace.db"
                prerace_db(d, db, chunk)
                r = subprocess.run(
                    [sys.executable, "-c", CHILD.format(here=str(HERE), db=str(db), d=d, out=str(part))],
                    capture_output=True, text=True)
            if r.returncode != 0 or not part.exists():
                ok = False
                print(f"  chunk {k} rc={r.returncode}", r.stderr[-400:], file=sys.stderr)
                break
            parts.append(part)
        if ok:
            import pandas as pd
            pd.concat([pd.read_csv(p) for p in parts]).sort_values(
                ["date", "course", "off", "rank_in_race"]).to_csv(out, index=False)
        for p in parts:
            p.unlink(missing_ok=True)
        status = "ok" if ok and out.exists() else "FAILED"
        if status != "ok":
            failures.append(d)
        print(f"[{i}/{len(todo)}] {d} {status} {time.time() - t0:.0f}s "
              f"(elapsed {(time.time() - t_all) / 60:.0f} min)", flush=True)
    (OUT_DIR / "failures.txt").write_text("\n".join(failures))
    print(f"done; failures: {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
