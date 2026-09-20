"""
Model picks vs bookmaker odds for one day (default today), Standard Racing API plan.

* Race not yet run: current bookmaker odds from /v1/racecards/standard (best price across the
  bookmakers, plus the market's own implied probability with the overround removed).
* Race finished: the closing odds (starting price, `sp_dec`) from /v1/results, plus who won.

Bettable = the offered price is at or above the break-even odds for the pick's confidence band:
break-even = 1 / precision of all backtest runners with blend_all >= the band's threshold
(predictions/backtest_wf, 115 days). See bettable_analysis.py for why that is necessary but not
sufficient. Writes predictions/odds_report_<date>.md.

    python today_odds_report.py [--date 2026-09-20] [--all-runners]
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date as date_cls
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "data_collection"))
import evaluate_backtest as ev  # noqa: E402
import fast_inference as fi  # noqa: E402
import inference as inf  # noqa: E402
from racingapi_client import RacingAPIClient, RacingAPIError  # noqa: E402

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def break_even_table() -> dict[float, float]:
    """{threshold: precision of backtest runners with blend_all >= threshold}."""
    df = ev.load(HERE / "predictions" / "backtest_wf")
    return {t: float(df.loc[df["blend_all"] >= t, "win"].mean()) for t in THRESHOLDS}


def break_even_odds(p: float, prec: dict[float, float]) -> float:
    ts = [t for t in THRESHOLDS if t <= p]
    return 1.0 / prec[ts[-1]] if ts else 1.0 / p


_SUFFIX = re.compile(r"\s*\([A-Za-z]{2,4}\)$")


def bare(name) -> str:
    """'Premier Tenor (FR)' -> 'premier tenor' (racecard horses carry no region suffix)."""
    return _SUFFIX.sub("", str(name)).strip().lower()


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def odds_frame(races: list[dict]) -> pd.DataFrame:
    rows = []
    for race in races:
        for r in race.get("runners", []):
            books = [o for o in r.get("odds") or [] if _num(o.get("decimal")) is not None]   # skip "SP" etc.
            dec = [float(o["decimal"]) for o in books]
            best = max(books, key=lambda o: float(o["decimal"])) if books else None
            rows.append({"race_id": race["race_id"], "hkey": bare(r.get("horse")),
                         "best_odds": max(dec) if dec else np.nan, "median_odds": float(np.median(dec)) if dec else np.nan,
                         "n_books": len(dec), "best_book": best["bookmaker"] if best else "",
                         "status": race.get("race_status"), "is_abandoned": race.get("is_abandoned")})
    df = pd.DataFrame(rows)
    imp = 1.0 / df["median_odds"]
    df["market_p"] = imp / imp.groupby(df["race_id"]).transform("sum")        # overround removed
    return df


def closing_frame(client: RacingAPIClient, day: date_cls) -> pd.DataFrame:
    rows = []
    for page in client.results_all_pages(day, day):
        for race in page.get("results", []):
            for x in race.get("runners", []):
                rows.append({"race_id": race["race_id"], "hkey": bare(x.get("horse")),
                             "sp_dec": pd.to_numeric(x.get("sp_dec"), errors="coerce"),
                             "finish": pd.to_numeric(x.get("position"), errors="coerce")})
    return pd.DataFrame(rows, columns=["race_id", "hkey", "sp_dec", "finish"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date_cls.today().isoformat())
    ap.add_argument("--all-runners", action="store_true", help="also list every runner with p >= 0.15")
    args = ap.parse_args()
    day = date_cls.fromisoformat(args.date)

    client = RacingAPIClient()
    cards = client.racecards_standard("today")
    odds = odds_frame(cards)
    try:
        closing = closing_frame(client, day)
    except RacingAPIError as e:
        print(f"(no closing odds: {str(e)[:120]})", file=sys.stderr)
        closing = pd.DataFrame(columns=["race_id", "hkey", "sp_dec", "finish"])

    preds = fi.predict_day(args.date, inf.load_live(args.date))
    preds["hkey"] = preds["horse"].map(bare)
    d = preds.merge(odds.drop_duplicates(["race_id", "hkey"]), on=["race_id", "hkey"], how="left") \
             .merge(closing.drop_duplicates(["race_id", "hkey"]), on=["race_id", "hkey"], how="left")
    assert len(d) == len(preds), "merge changed the number of runners"
    unmatched = int(d["best_odds"].isna().sum())
    prec = break_even_table()

    d["finished"] = d.groupby("race_id")["sp_dec"].transform(lambda s: s.notna().any())
    d["price"] = np.where(d["finished"], d["sp_dec"], d["best_odds"])          # closing if run, else best available
    d["price_kind"] = np.where(d["finished"], "closing SP", "best now")
    d["break_even"] = d["blend_all"].map(lambda p: break_even_odds(p, prec))
    d["fair_odds"] = 1 / d["blend_all"]
    d["bettable"] = d["price"] >= d["break_even"]
    d["edge"] = d["blend_all"] * d["price"] - 1                                 # model p x price - 1
    d["won"] = np.where(d["finished"], d["finish"] == 1, np.nan)

    top = d[d["rank_in_race"] == 1].sort_values(["course", "off"])
    show = top[["course", "off", "horse", "blend_all", "fair_odds", "price", "price_kind", "break_even", "bettable",
                "edge", "market_p", "n_books", "won"]].copy()
    show["off"] = show["off"].astype(str)
    lines = [f"# Model picks vs bookmaker odds, {args.date}",
             f"{d['race_id'].nunique()} races, {len(d)} runners ({unmatched} without odds). Finished races use the closing SP, "
             f"the rest the best price across bookmakers right now. Break-even odds = 1 / backtest precision of runners at or above "
             f"the pick's confidence band ({', '.join(f'>={t:.2f}: {1 / prec[t]:.2f}' for t in THRESHOLDS)}).\n",
             "## #1 pick per race",
             ev.md(show.rename(columns={"blend_all": "p", "price": "odds", "break_even": "BE_odds"}).reset_index(drop=True), "{:.3f}"), ""]
    n = len(top)
    lines += [f"**#1 picks at a bettable price: {int(top['bettable'].sum())} of {n} ({top['bettable'].mean():.0%})**; "
              f"with model edge p x odds - 1 >= 0: {int((top['edge'] >= 0).sum())} of {n}.",
              f"Races finished so far: {int(top['finished'].sum())} of {n}."]
    fin = top[top["finished"]]
    if len(fin):
        lines.append(f"Finished: #1 picks won {int(fin['won'].sum())} of {len(fin)}; among bettable ones "
                     f"{int(fin[fin['bettable']]['won'].sum())} of {int(fin['bettable'].sum())}.")
    if args.all_runners:
        allr = d[d["blend_all"] >= 0.15].sort_values(["course", "off", "rank_in_race"])
        lines += ["", "## Every runner with p >= 0.15",
                  ev.md(allr[["course", "off", "horse", "blend_all", "price", "break_even", "bettable", "edge"]].reset_index(drop=True), "{:.3f}")]
    report = "\n".join(lines)
    (HERE / "predictions" / f"odds_report_{args.date}.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
