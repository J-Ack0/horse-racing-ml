"""
Parity check: fast_inference's features vs the ACTUAL training matrix (cache/features.pkl).

For a few historical days, blank the results, rebuild the 135 features with the fast path
(history strictly before that day, from 2021-01-01) and compare every feature to the training
matrix row for the same runner. Verified 2026-09-20: 0 of 135 features differ on 2026-05-20,
2025-10-04 and 2024-06-15 (every row matched).

    python ml/kaggle_v2/verify_fast_vs_training.py            # from the repo/worktree root
"""
import sys, time, sqlite3, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'ml/kaggle_v2')
import numpy as np, pandas as pd
import fast_inference as fi, inference as inf
pkl = pd.read_pickle('ml/kaggle_v2/cache/features.pkl')
feats = [l.strip() for l in open('ml/kaggle_v2/cache/feature_names.txt') if l.strip()]
print("pkl", pkl.shape, "date range", pkl.date.min(), pkl.date.max(), "| has num:", 'num' in pkl.columns)
con = sqlite3.connect('file:data_ext/raceform.db?mode=ro', uri=True)
for d in ['2026-05-20', '2025-10-04', '2024-06-15']:
    t0 = time.time()
    rows = pd.read_sql(f"SELECT {inf.RAW_COLS} FROM data WHERE date=? AND {fi.COURSE_FILTER}", con, params=[d])
    fast = fi.features_for_day(d, rows)
    dt = time.time() - t0
    ref = pkl[pkl.date == pd.Timestamp(d)]
    m = fast.merge(ref, on=['race_id', 'num'], suffixes=('_f', '_p'))
    bad = []
    for f in [x for x in feats if x != 'num']:
        a = m[f + '_f'].astype(np.float32).to_numpy(); b = m[f + '_p'].to_numpy()
        ok = (a == b) | (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-4, atol=1e-6, equal_nan=True)
        if not ok.all(): bad.append((f, int((~ok).sum()), float(np.nanmax(np.abs(a - b)))))
    print(f"{d}: fast rows {len(fast)} ref rows {len(ref)} matched {len(m)} in {dt:.1f}s | features differing: {len(bad)}/{len(feats)}", flush=True)
    for f, n, mx in sorted(bad, key=lambda x: -x[1])[:12]: print(f"    {f}: {n} rows, max abs diff {mx:.4g}")
