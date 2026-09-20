"""
Settle a saved each-way Lucky 15 slip from /v1/results and push the outcome to the phone.

Waits (polls every few minutes) until all legs have a result, then sends ONE note through
notify-phone.sh with each leg's finishing position and SP and the slip's net return at the PLANNED
prices/terms in the slip file (your actual prices may differ). If some races are still not in the
results at the deadline it sends a [Status] note with what is known.

    python check_selections.py slips/2026-09-20_lucky15.json [--wait-minutes 120] [--poll-seconds 300] [--no-notify]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "data_collection"))

_SUFFIX = re.compile(r"\s*\([A-Za-z]{2,4}\)$")
NOTIFY = Path.home() / ".local" / "bin" / "notify-phone.sh"


def bare(name) -> str:
    return _SUFFIX.sub("", str(name)).strip().lower()


def find_leg(races: list[dict], leg: dict) -> dict:
    """{'state': 'pending'|'nr'|'done', 'pos': int|str|None, 'sp': float|None} for one leg."""
    for race in races:
        if not str(race.get("course", "")).lower().startswith(leg["course"].lower()):
            continue
        if (race.get("off_dt") or "")[11:16] != leg["off"]:
            continue
        for r in race.get("runners", []):
            if bare(r.get("horse")) == bare(leg["horse"]):
                pos_raw = r.get("position")
                try:
                    pos = int(float(pos_raw))
                except (TypeError, ValueError):
                    pos = str(pos_raw).strip() or None
                try:
                    sp = float(r.get("sp_dec"))
                except (TypeError, ValueError):
                    sp = None
                return {"state": "done", "pos": pos, "sp": sp}
        return {"state": "nr", "pos": None, "sp": None}     # race is in the results but the horse is not: withdrawn
    return {"state": "pending", "pos": None, "sp": None}


def leg_outcome(leg: dict, res: dict) -> str:
    if res["state"] == "pending":
        return "pending"
    if res["state"] == "nr":
        return "NR"
    pos = res["pos"]
    if pos == 1:
        return "W"
    if isinstance(pos, int) and pos <= leg["places"]:
        return "P"
    return "L"


def settle(slip: dict, results: list[dict]) -> dict:
    """Net return of the each-way Lucky 15 (15 win + 15 place lines) at the slip's prices and terms."""
    legs, outs, ress = slip["legs"], [], []
    for leg in legs:
        res = find_leg(results, leg)
        ress.append(res)
        outs.append(leg_outcome(leg, res))
    done = all(o != "pending" for o in outs)
    u = float(slip["unit_stake"])
    win_r, place_r = [], []
    for leg, o in zip(legs, outs):
        d = float(leg["price"]); dp = 1 + (d - 1) / leg["denom"]
        # per-leg return multiplier of a single line: winner pays d (win) / dp (place); placed pays dp on the place
        # line only; a withdrawn horse makes the leg void (multiplier 1, the stake rolls on).
        win_r.append(d if o == "W" else 1.0 if o == "NR" else 0.0)
        place_r.append(dp if o in ("W", "P") else 1.0 if o == "NR" else 0.0)
    win_part = float(np.prod([1 + r for r in win_r]) - 1)
    place_part = float(np.prod([1 + r for r in place_r]) - 1)
    stake = u * 30
    returns = u * (win_part + place_part)
    return {"done": done, "outcomes": outs, "results": ress, "stake": stake, "returns": returns, "net": returns - stake}


def fmt_pos(res: dict) -> str:
    if res["state"] == "pending":
        return "not run yet"
    if res["state"] == "nr":
        return "non-runner (leg void)"
    p = res["pos"]
    sp = f", SP {res['sp']:.2f}" if res["sp"] else ""
    if isinstance(p, int):
        return f"{p}{'st' if p == 1 else 'nd' if p == 2 else 'rd' if p == 3 else 'th'}{sp}"
    return f"{p or 'no position'}{sp}"


def fetch_results(day: date) -> list[dict]:
    from racingapi_client import RacingAPIClient
    races: list[dict] = []
    for page in RacingAPIClient().results_all_pages(day, day):
        races.extend(page.get("results", []))
    return races


def notify(kind: str, line: str, body: str) -> None:
    subprocess.run([str(NOTIFY), kind, line, body], check=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slip")
    ap.add_argument("--wait-minutes", type=int, default=120)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--no-notify", action="store_true")
    args = ap.parse_args()
    slip = json.load(open(args.slip))
    day = date.fromisoformat(slip["date"])
    cur = slip.get("currency", "EUR")

    deadline = time.time() + args.wait_minutes * 60
    while True:
        try:
            out = settle(slip, fetch_results(day))
        except Exception as e:  # noqa: BLE001 - transient API errors: keep polling until the deadline
            out = None
            print(f"fetch failed: {e}", file=sys.stderr)
        if (out and out["done"]) or time.time() >= deadline:
            break
        time.sleep(args.poll_seconds)

    if out is None:
        line, body, kind = f"{slip['name']} {slip['date']}: could not fetch results", "The results call kept failing.", "error"
    else:
        legs = [f"{leg['horse']} ({leg['course']} {leg['off']}): {fmt_pos(r)}  [{o}]"
                for leg, r, o in zip(slip["legs"], out["results"], out["outcomes"])]
        n_win = out["outcomes"].count("W"); n_pl = out["outcomes"].count("P")
        net = out["net"]
        if out["done"]:
            line = f"{slip['name']} {slip['date']}: {n_win} won, {n_pl} placed, net {net:+.2f} {cur} (returns {out['returns']:.2f} on {out['stake']:.2f})"
            kind = "done"
        else:
            line = f"{slip['name']} {slip['date']}: results incomplete at the deadline ({out['outcomes'].count('pending')} race(s) not in yet)"
            kind = "status"
        body = "\n".join(legs) + f"\nAt the planned prices/terms in the slip; your actual prices may differ.\nW = won, P = placed, L = lost, NR = non-runner (leg void)."
    print(line + "\n" + body)
    if not args.no_notify:
        notify(kind, line, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
