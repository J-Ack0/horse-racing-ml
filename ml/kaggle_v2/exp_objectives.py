"""Experiment 1: does race-conditional (group-structured) modelling beat independent
binary classification? Also: native-NaN vs median imputation.

All models share the same features, same chronological split, same base hyperparameters
(the ones inherited from the old model) so the objective is the only thing that varies.
"""
import sys, json, time
import numpy as np
import pandas as pd
import xgboost as xgb
from common import (HERE, load, chrono_split, race_metrics, ece, make_group_index,
                    softmax_by_group, race_softmax_obj)

IRE = '--ire' in sys.argv
SEED = 7
BASE = dict(max_depth=5, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=1, nthread=4, seed=SEED)
NROUND, ESR = 3000, 50


def prep(ire_only=False):
    df, feats = load(ire_only)
    tr, va, te, bounds = chrono_split(df)
    print(f'split dates {bounds}  train={tr.sum()} val={va.sum()} test={te.sum()}', flush=True)
    X = df[feats].to_numpy(dtype=np.float32)
    y = df['win'].to_numpy(dtype=np.float32)
    return df, feats, X, y, tr, va, te


def median_fill(X, tr):
    med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xf = np.where(np.isnan(X), med, X)
    return Xf


def dmat(X, y, idx, group_ids=None):
    d = xgb.DMatrix(X[idx], label=y[idx], missing=np.nan)
    if group_ids is not None:
        _, counts = np.unique(group_ids[idx], return_counts=True)
        # group sizes must follow row order; race_ids are contiguous after date sort
        sizes = pd.Series(group_ids[idx]).groupby(
            pd.Series(group_ids[idx]), sort=False).size().to_numpy()
        d.set_group(sizes)
    return d


def eval_model(name, df, te, y, score, pn, rows):
    m = race_metrics(df[te], y[te], score, pn)
    m['ece'] = ece(y[te], np.clip(pn, 0, 1))
    m['model'] = name
    rows.append(m)
    print(f"{name:28s} AUC={m['auc']:.4f} AP={m['ap']:.4f} top1={m['top1']:.4f} "
          f"MRR={m['mrr']:.4f} ndcg3={m['ndcg3']:.4f} rll={m['race_logloss']:.4f} "
          f"brierN={m['brier_norm']:.5f} ECE={m['ece']:.4f}", flush=True)


def norm_probs(df, te, p):
    s = pd.Series(p).groupby(df.loc[te, 'race_id'].to_numpy()).transform('sum')
    return (p / s.to_numpy())


def main():
    df, feats, X, y, tr, va, te = prep(IRE)
    rid = df['race_id'].to_numpy()
    rows = []

    # ---------------- A. binary logistic, median imputation (replicates baseline) -----
    Xf = median_fill(X, tr)
    dtr = xgb.DMatrix(Xf[tr], label=y[tr]); dva = xgb.DMatrix(Xf[va], label=y[va])
    dte = xgb.DMatrix(Xf[te])
    p = dict(BASE, objective='binary:logistic', eval_metric='auc')
    t = time.time()
    b1 = xgb.train(p, dtr, NROUND, evals=[(dva, 'val')], early_stopping_rounds=ESR,
                   verbose_eval=False)
    s = b1.predict(dte, iteration_range=(0, b1.best_iteration + 1))
    print(f'[binary/median] best_iter={b1.best_iteration} ({time.time()-t:.0f}s)', flush=True)
    eval_model('binary median-impute', df, te, y, s, norm_probs(df, te, s), rows)

    # ---------------- B. binary logistic, native NaN ---------------------------------
    dtr = xgb.DMatrix(X[tr], label=y[tr], missing=np.nan)
    dva = xgb.DMatrix(X[va], label=y[va], missing=np.nan)
    dte = xgb.DMatrix(X[te], missing=np.nan)
    b2 = xgb.train(p, dtr, NROUND, evals=[(dva, 'val')], early_stopping_rounds=ESR,
                   verbose_eval=False)
    s2 = b2.predict(dte, iteration_range=(0, b2.best_iteration + 1))
    print(f'[binary/nan] best_iter={b2.best_iteration}', flush=True)
    eval_model('binary native-NaN', df, te, y, s2, norm_probs(df, te, s2), rows)

    # ---------------- C. XGBoost learning-to-rank objectives -------------------------
    def group_sizes(mask):
        r = rid[mask]
        return pd.Series(r).groupby(pd.Series(r), sort=False).size().to_numpy()

    for obj, metric in [('rank:pairwise', 'auc'), ('rank:ndcg', 'ndcg@3'),
                        ('rank:map', 'map@3')]:
        dtrg = xgb.DMatrix(X[tr], label=y[tr], missing=np.nan)
        dtrg.set_group(group_sizes(tr))
        dvag = xgb.DMatrix(X[va], label=y[va], missing=np.nan)
        dvag.set_group(group_sizes(va))
        pr = dict(BASE, objective=obj, eval_metric=metric)
        try:
            br = xgb.train(pr, dtrg, NROUND, evals=[(dvag, 'val')],
                           early_stopping_rounds=ESR, verbose_eval=False)
        except Exception as e:
            print('skip', obj, e); continue
        sr = br.predict(dte, iteration_range=(0, br.best_iteration + 1))
        gi_te = make_group_index(rid[te])
        pn = softmax_by_group(sr.astype(np.float64), gi_te, gi_te.max() + 1)
        print(f'[{obj}] best_iter={br.best_iteration}', flush=True)
        eval_model(obj, df, te, y, sr, pn, rows)

    # ---------------- D. custom race-softmax (conditional logit) ---------------------
    gi_tr = make_group_index(rid[tr]); gi_va = make_group_index(rid[va])
    gi_te = make_group_index(rid[te])
    ytr, yva = y[tr].astype(np.float64), y[va].astype(np.float64)
    obj = race_softmax_obj(gi_tr, gi_tr.max() + 1, ytr)

    def val_metric(preds, dm):
        pv = softmax_by_group(preds.astype(np.float64), gi_va, gi_va.max() + 1)
        ll = -np.log(np.clip(pv[yva == 1], 1e-12, None)).mean()
        return 'race_logloss', float(ll)

    ps = dict(BASE, objective='reg:squarederror', base_score=0.0, disable_default_eval_metric=1)
    dtr = xgb.DMatrix(X[tr], label=y[tr], missing=np.nan)
    dva = xgb.DMatrix(X[va], label=y[va], missing=np.nan)
    t = time.time()
    bs = xgb.train(ps, dtr, NROUND, evals=[(dva, 'val')], obj=obj,
                   custom_metric=val_metric, early_stopping_rounds=ESR,
                   maximize=False, verbose_eval=False)
    ss = bs.predict(dte, iteration_range=(0, bs.best_iteration + 1)).astype(np.float64)
    pn = softmax_by_group(ss, gi_te, gi_te.max() + 1)
    print(f'[race-softmax] best_iter={bs.best_iteration} ({time.time()-t:.0f}s)', flush=True)
    eval_model('race-softmax (cond. logit)', df, te, y, ss, pn, rows)
    bs.save_model(HERE + '/cache/softmax_base%s.json' % ('_ire' if IRE else ''))

    out = pd.DataFrame(rows)[['model', 'auc', 'ap', 'top1', 'top3', 'mrr', 'ndcg3',
                              'race_logloss', 'brier_norm', 'ece']]
    tag = '_ire' if IRE else ''
    out.to_csv(HERE + f'/results_objectives{tag}.csv', index=False)
    print('\n', out.to_string(index=False))


if __name__ == '__main__':
    main()
