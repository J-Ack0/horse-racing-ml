"""Fair retry of XGBoost's built-in LambdaMART objectives.

In the first pass rank:ndcg / rank:map early-stopped at iteration 0: with binary relevance
and only one relevant document per group, NDCG@3 / MAP@3 on the validation block is a very
coarse, plateau-heavy statistic, so a patience of 50 rounds fired before it ever moved.
Here they get: a proper learning rate, much more patience, `lambdarank_pair_method=mean`
with several sampled pairs per query, and early stopping on race log-loss (the same
continuous criterion the other models use) so the comparison is like-for-like.
"""
import numpy as np, pandas as pd, xgboost as xgb, sys, time
from common import (HERE, load, chrono_split, race_metrics, ece, make_group_index,
                    softmax_by_group)

IRE = '--ire' in sys.argv
df, feats = load(IRE)
X = df[feats].to_numpy(np.float32); y = df['win'].to_numpy(np.float32)
rid = df['race_id'].to_numpy()
tr, va, te, bounds = chrono_split(df)
gi_va = make_group_index(rid[va]); gi_te = make_group_index(rid[te])
yva = y[va].astype(np.float64)


def sizes(mask):
    r = pd.Series(rid[mask])
    return r.groupby(r, sort=False).size().to_numpy()


dtr = xgb.DMatrix(X[tr], label=y[tr], missing=np.nan); dtr.set_group(sizes(tr))
dva = xgb.DMatrix(X[va], label=y[va], missing=np.nan); dva.set_group(sizes(va))
dte = xgb.DMatrix(X[te], missing=np.nan)


def rll_metric(preds, dm):
    pv = softmax_by_group(preds.astype(np.float64), gi_va, gi_va.max() + 1)
    return 'rll', float(-np.log(np.clip(pv[yva == 1], 1e-12, None)).mean())


rows = []
for obj, extra in [('rank:ndcg', dict(lambdarank_pair_method='mean',
                                      lambdarank_num_pair_per_sample=8,
                                      ndcg_exp_gain=False)),
                   ('rank:pairwise', dict(lambdarank_pair_method='mean',
                                          lambdarank_num_pair_per_sample=8))]:
    p = dict(extra, objective=obj, max_depth=6, learning_rate=0.05, subsample=0.9,
             colsample_bytree=0.8, min_child_weight=5, nthread=4, seed=7,
             disable_default_eval_metric=1)
    t = time.time()
    b = xgb.train(p, dtr, 1200, evals=[(dva, 'val')], custom_metric=rll_metric,
                  early_stopping_rounds=120, maximize=False, verbose_eval=False)
    s = b.predict(dte, iteration_range=(0, b.best_iteration + 1)).astype(np.float64)
    pn = softmax_by_group(s, gi_te, gi_te.max() + 1)
    m = race_metrics(df[te], y[te], s, pn); m['ece'] = ece(y[te], pn); m['model'] = obj + ' (retry)'
    rows.append(m)
    print(f"{obj} it={b.best_iteration} ({time.time()-t:.0f}s) AUC={m['auc']:.4f} "
          f"top1={m['top1']:.4f} MRR={m['mrr']:.4f} rll={m['race_logloss']:.4f}", flush=True)

out = pd.DataFrame(rows)[['model', 'auc', 'ap', 'top1', 'top3', 'mrr', 'ndcg3',
                          'race_logloss', 'brier_norm', 'ece']]
out.to_csv(HERE + ('/results_rank_retry%s.csv' % ('_ire' if IRE else '')), index=False)
print(out.to_string(index=False))
