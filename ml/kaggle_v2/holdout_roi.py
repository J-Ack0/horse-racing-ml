"""
Price the model's held-out picks at their real starting prices (SP).

Loads the held-out test slice (same chronological split as exp_final.py),
scores it with the saved base blend_all models, joins each runner's SP from
data_ext/raceform.db on (race_id, num), and reports flat-stake ROI for:
the model's #1 pick, the market favourite, every runner, SP bands, minimum-
price filters, and "value" filters (p_model * SP >= 1 + margin).

Conventions: 1 unit stake, decimal odds INCLUDE the stake (SP 5/2 -> 3.5),
ROI = mean(win * odds - 1), no commission, no tax, SP as a stand-in for an
obtainable price (it is not: see docs/PROJECT_NOTES.md, section 6.4).

Needs ~3 GB RAM and about two minutes on the Pi. Run from the repo root:
    venv/bin/python ml/kaggle_v2/holdout_roi.py
"""
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from common import load, chrono_split, softmax_by_group, make_group_index  # noqa: E402

df, feats = load(False)
tr, va, te, (d1, d2) = chrono_split(df)
T = df[te].reset_index(drop=True)
print("test window starts", d2, " races", T.race_id.nunique(), " runners", len(T))

X = xgb.DMatrix(T[feats].to_numpy(np.float32), missing=np.nan)
rid = T.race_id.to_numpy()
g = make_group_index(rid)
ng = g.max() + 1
pn = {}
for name, f, kind in [("binary", "final_binary.json", "prob"), ("softmax", "final_softmax.json", "score"),
                      ("pl", "final_pltop3.json", "score")]:
    b = xgb.Booster()
    b.load_model(str(HERE / "cache" / f))
    raw = b.predict(X)
    if kind == "score":
        pn[name] = softmax_by_group(raw.astype(np.float64), g, ng)
    else:
        s = pd.Series(raw, dtype=np.float64).groupby(rid).transform("sum").to_numpy()
        pn[name] = raw.astype(np.float64) / s
lb = sum(np.log(np.clip(v, 1e-12, None)) for v in pn.values()) / 3
T["p"] = softmax_by_group(lb, g, ng)

con = sqlite3.connect(ROOT / "data_ext" / "raceform.db")
sp = pd.read_sql(f"SELECT race_id, num, sp FROM data WHERE date >= '{pd.Timestamp(d2).date()}' AND date != 'date'", con)


def dec(s):
    s = str(s).upper()
    if "EVS" in s or "EVENS" in s:
        return 2.0
    m = re.search(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) / float(m.group(2)) + 1 if m else np.nan


sp["odds"] = sp.sp.map(dec)
sp["num"] = pd.to_numeric(sp.num, errors="coerce")
sp["race_id"] = pd.to_numeric(sp.race_id, errors="coerce")
T["race_id_n"] = pd.to_numeric(T.race_id, errors="coerce")
T["num"] = pd.to_numeric(T.num, errors="coerce")
M = T.merge(sp[["race_id", "num", "odds"]], left_on=["race_id_n", "num"], right_on=["race_id", "num"],
            how="left", suffixes=("", "_sp"))
print("runners with parsed SP:", int(M.odds.notna().sum()), "of", len(M))

M["win"] = M.win.astype(int)
ok = M.groupby("race_id_n").odds.transform(lambda s: s.notna().all())
M = M[ok].copy()
print("races with a full SP book:", M.race_id_n.nunique())
M["imp"] = 1 / M.odds
M["imp_n"] = M.imp / M.groupby("race_id_n").imp.transform("sum")
print("mean overround (sum of 1/SP per race): %.3f" % M.groupby("race_id_n").imp.sum().mean())


def stats(sel, label):
    n = len(sel)
    w = sel.win.mean()
    roi = (sel.win * sel.odds - 1).mean()
    print(f"{label:38} bets={n:5d} win={w:.3f} mean_odds={sel.odds.mean():5.2f} ROI={roi:+.3f}")


top = M.loc[M.groupby("race_id_n").p.idxmax()]
fav = M.loc[M.groupby("race_id_n").odds.idxmin()]
stats(top, "model #1 pick at SP")
stats(fav, "market favourite at SP")
stats(M, "every runner at SP")
print("\nmodel #1 pick, by SP band")
for lo, hi in [(1, 2.5), (2.5, 3.5), (3.5, 5), (5, 8), (8, 200)]:
    stats(top[(top.odds >= lo) & (top.odds < hi)], f"  SP {lo}-{hi}")
print("\nmodel #1 pick, only where SP >= threshold")
for t in [3.0, 3.67, 4.0, 5.0, 6.0]:
    stats(top[top.odds >= t], f"  SP >= {t}")
print("\nvalue filter: bet #1 pick only if p_model*SP >= 1+margin")
for mg in [0.0, 0.1, 0.25, 0.5]:
    stats(top[top.p * top.odds >= 1 + mg], f"  margin {mg:.2f}")
print("\nvalue filter over ALL runners: bet any runner with p_model*SP >= 1+margin")
for mg in [0.1, 0.25, 0.5]:
    stats(M[M.p * M.odds >= 1 + mg], f"  margin {mg:.2f}")
print("\nmodel vs market: mean p_model on winners %.3f vs market-implied %.3f"
      % (M[M.win == 1].p.mean(), M[M.win == 1].imp_n.mean()))
print("top pick: mean p_model %.3f, market-implied prob of same horse %.3f, actual win %.3f"
      % (top.p.mean(), top.imp_n.mean(), top.win.mean()))
