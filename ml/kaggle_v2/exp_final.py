"""Experiment 3: final models, Plackett-Luce top-K, ensembling and calibration.

Trains (on train, early-stopped on val) the tuned binary and race-softmax models plus a
Plackett-Luce top-3 model, blends them, checks calibration, applies isotonic
recalibration fitted on the validation block only, and reports everything on the held-out
final 20% of dates.

Usage: exp_final.py [--ire]
"""
import sys, json, os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from common import (HERE, load, chrono_split, race_metrics, ece, make_group_index,
                    softmax_by_group, race_softmax_obj, plackett_luce_topk_obj)

IRE = '--ire' in sys.argv
TAG = '_ire' if IRE else ''
NROUND, ESR = 3000, 60


def load_hp(name):
    f = HERE + f'/best_params{TAG}_{name}.json'
    if os.path.exists(f):
        hp = json.load(open(f))
        hp.pop('_mean_iter', None)
        return hp
    print('!! no tuned params for', name, '- using inherited defaults')
    return dict(max_depth=5, min_child_weight=1, subsample=0.8, colsample_bytree=0.8)


def rank_within_race(df):
    """Finishing rank 1..n; non-finishers ranked after all finishers."""
    pos = df['pos_num'].to_numpy()
    key = np.where(np.isnan(pos), 1e6, pos)
    s = pd.Series(key)
    return s.groupby(df['race_id'].to_numpy(), sort=False).rank(method='first').to_numpy()


def main():
    df, feats = load(IRE)
    X = df[feats].to_numpy(np.float32)
    y = df['win'].to_numpy(np.float32)
    rid = df['race_id'].to_numpy()
    rnk = rank_within_race(df)
    tr, va, te, bounds = chrono_split(df)
    print(f'ire={IRE} split={bounds} n={len(df)} feats={len(feats)}', flush=True)

    gi = {k: make_group_index(rid[m]) for k, m in [('tr', tr), ('va', va), ('te', te)]}
    ng = {k: v.max() + 1 for k, v in gi.items()}
    yv = {k: y[m].astype(np.float64) for k, m in [('tr', tr), ('va', va), ('te', te)]}

    dtr = xgb.DMatrix(X[tr], label=y[tr], missing=np.nan)
    dva = xgb.DMatrix(X[va], label=y[va], missing=np.nan)
    dte = xgb.DMatrix(X[te], missing=np.nan)

    preds = {}   # name -> (val_score, test_score, kind)

    def custom_metric_factory():
        def m(p, dm):
            pv = softmax_by_group(p.astype(np.float64), gi['va'], ng['va'])
            return 'rll', float(-np.log(np.clip(pv[yv['va'] == 1], 1e-12, None)).mean())
        return m

    # ---- 1. tuned binary ----
    hp = load_hp('binary')
    p = dict(hp, objective='binary:logistic', eval_metric='logloss',
             learning_rate=0.03, nthread=4, seed=7)
    b = xgb.train(p, dtr, NROUND, evals=[(dva, 'val')], early_stopping_rounds=ESR,
                  verbose_eval=False)
    print('binary best_iter', b.best_iteration, flush=True)
    preds['binary'] = (b.predict(dva, iteration_range=(0, b.best_iteration + 1)),
                       b.predict(dte, iteration_range=(0, b.best_iteration + 1)), 'prob')
    b.save_model(HERE + f'/cache/final_binary{TAG}.json')

    # ---- 2. tuned race-softmax ----
    hp = load_hp('softmax')
    ps = dict(hp, objective='reg:squarederror', base_score=0.0,
              disable_default_eval_metric=1, learning_rate=0.03, nthread=4, seed=7)
    obj = race_softmax_obj(gi['tr'], ng['tr'], yv['tr'])
    bs = xgb.train(ps, dtr, NROUND, evals=[(dva, 'val')], obj=obj,
                   custom_metric=custom_metric_factory(), early_stopping_rounds=ESR,
                   maximize=False, verbose_eval=False)
    print('softmax best_iter', bs.best_iteration, flush=True)
    preds['softmax'] = (bs.predict(dva, iteration_range=(0, bs.best_iteration + 1)),
                        bs.predict(dte, iteration_range=(0, bs.best_iteration + 1)), 'score')
    bs.save_model(HERE + f'/cache/final_softmax{TAG}.json')

    # ---- 3. Plackett-Luce top-3 ----
    objpl = plackett_luce_topk_obj(gi['tr'], ng['tr'], rnk[tr], K=3)
    bp = xgb.train(ps, dtr, NROUND, evals=[(dva, 'val')], obj=objpl,
                   custom_metric=custom_metric_factory(), early_stopping_rounds=ESR,
                   maximize=False, verbose_eval=False)
    print('PL top3 best_iter', bp.best_iteration, flush=True)
    preds['pl_top3'] = (bp.predict(dva, iteration_range=(0, bp.best_iteration + 1)),
                        bp.predict(dte, iteration_range=(0, bp.best_iteration + 1)), 'score')
    bp.save_model(HERE + f'/cache/final_pltop3{TAG}.json')

    # ---- normalise everything to per-race probabilities ----
    def to_pn(v, split, kind):
        v = np.asarray(v, dtype=np.float64)
        if kind == 'score':
            return softmax_by_group(v, gi[split], ng[split])
        s = pd.Series(v).groupby(rid[{'va': va, 'te': te}[split]]).transform('sum').to_numpy()
        return v / s

    pn_va = {k: to_pn(v[0], 'va', v[2]) for k, v in preds.items()}
    pn_te = {k: to_pn(v[1], 'te', v[2]) for k, v in preds.items()}

    # ---- 4. ensembles (log-prob average = geometric mean, then renormalise) ----
    def blend(names, w=None):
        w = w or [1 / len(names)] * len(names)
        lv = sum(wi * np.log(np.clip(pn_va[n], 1e-12, None)) for wi, n in zip(w, names))
        lt = sum(wi * np.log(np.clip(pn_te[n], 1e-12, None)) for wi, n in zip(w, names))
        return (softmax_by_group(lv, gi['va'], ng['va']),
                softmax_by_group(lt, gi['te'], ng['te']))
    pn_va['blend_bin_sm'], pn_te['blend_bin_sm'] = blend(['binary', 'softmax'])
    pn_va['blend_all'], pn_te['blend_all'] = blend(['binary', 'softmax', 'pl_top3'])

    # ---- 5. isotonic recalibration fitted on VALIDATION only ----
    best_for_cal = 'blend_bin_sm'
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso.fit(pn_va[best_for_cal], yv['va'])
    cal = np.clip(iso.predict(pn_te[best_for_cal]), 1e-9, 1)
    cal = cal / pd.Series(cal).groupby(rid[te]).transform('sum').to_numpy()
    pn_te[best_for_cal + '+isotonic'] = cal

    # ---- 6. report ----
    rows = []
    dte_idx = df[te]
    for name, pn in pn_te.items():
        m = race_metrics(dte_idx, y[te], pn, pn)
        m['ece'] = ece(y[te], pn)
        m['model'] = name
        rows.append(m)
    out = pd.DataFrame(rows)[['model', 'auc', 'ap', 'top1', 'top3', 'mrr', 'ndcg3',
                              'race_logloss', 'brier_norm', 'ece']]
    out.to_csv(HERE + f'/results_final{TAG}.csv', index=False)
    print('\n', out.to_string(index=False), flush=True)

    # ---- 7. if UK+IRE model, also score its Irish subset for a like-for-like check ----
    if not IRE and (dte_idx['is_ire'] == 1).any():
        ire_mask = (dte_idx['is_ire'] == 1).to_numpy()
        sub = dte_idx[ire_mask]
        rows2 = []
        for name in ['binary', 'softmax', 'blend_bin_sm']:
            pv = pn_te[name][ire_mask]
            pv = pv / pd.Series(pv).groupby(sub['race_id'].to_numpy()).transform('sum').to_numpy()
            m = race_metrics(sub, y[te][ire_mask], pv, pv)
            m['model'] = name + ' [IRE subset of UK+IRE model]'
            rows2.append(m)
        o2 = pd.DataFrame(rows2)[['model', 'auc', 'ap', 'top1', 'mrr', 'race_logloss']]
        o2.to_csv(HERE + '/results_ire_subset.csv', index=False)
        print('\n', o2.to_string(index=False))

    # ---- 8. reliability table for the headline model ----
    d = pd.DataFrame({'p': pn_te[best_for_cal], 'y': y[te]})
    d['b'] = pd.qcut(d['p'].rank(method='first'), 12, labels=False)
    rel = d.groupby('b').agg(n=('y', 'size'), mean_p=('p', 'mean'), obs=('y', 'mean'))
    rel.to_csv(HERE + f'/reliability{TAG}.csv')
    print('\nreliability (blend, race-normalised):\n', rel.to_string())
    dc = pd.DataFrame({'p': pn_te[best_for_cal + '+isotonic'], 'y': y[te]})
    dc['b'] = pd.qcut(dc['p'].rank(method='first'), 12, labels=False)
    print('\nreliability (isotonic):\n',
          dc.groupby('b').agg(n=('y', 'size'), mean_p=('p', 'mean'), obs=('y', 'mean')).to_string())


if __name__ == '__main__':
    main()
