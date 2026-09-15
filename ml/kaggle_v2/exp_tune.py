"""Experiment 2: real hyperparameter search, chronologically honest.

Stage 1: random search over a sensible XGBoost space, scored on the single chronological
         validation block (fast screen, learning_rate fixed at 0.05 to keep runs short).
Stage 2: the top configurations are re-scored with rolling-origin (walk-forward) CV --
         expanding train window, contiguous later validation block -- which is the
         time-series-correct analogue of k-fold and gives a less split-lucky estimate.

Never shuffles across time; no random k-fold anywhere.

Usage: exp_tune.py [--ire] [--obj softmax|binary] [--n 24]
"""
import sys, json, time
import numpy as np
import pandas as pd
import xgboost as xgb
from common import (HERE, load, chrono_split, walk_forward_folds, race_metrics,
                    make_group_index, softmax_by_group, race_softmax_obj)

IRE = '--ire' in sys.argv
OBJ = sys.argv[sys.argv.index('--obj') + 1] if '--obj' in sys.argv else 'softmax'
NTRIAL = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 12
LR = 0.08
NROUND, ESR = 900, 30
rng = np.random.default_rng(11)


def sample_params():
    return dict(
        max_depth=int(rng.choice([3, 4, 5, 6, 7, 8])),
        min_child_weight=float(rng.choice([1, 3, 5, 10, 20, 50])),
        subsample=float(rng.uniform(0.6, 1.0)),
        colsample_bytree=float(rng.uniform(0.4, 1.0)),
        colsample_bylevel=float(rng.choice([0.6, 0.8, 1.0])),
        reg_lambda=float(10 ** rng.uniform(-1, 2)),
        reg_alpha=float(10 ** rng.uniform(-3, 1)),
        gamma=float(10 ** rng.uniform(-3, 0.5)) if rng.random() < 0.5 else 0.0,
    )


def fit_eval(X, y, rid, tr, va, hp, obj=OBJ, nround=NROUND, esr=ESR, lr=LR, seed=7):
    """Train on tr, early-stop on va. Returns (val race_logloss, val auc, best_iter, booster)."""
    p = dict(hp, learning_rate=lr, nthread=4, seed=seed)
    dtr = xgb.DMatrix(X[tr], label=y[tr], missing=np.nan)
    dva = xgb.DMatrix(X[va], label=y[va], missing=np.nan)
    gi_va = make_group_index(rid[va]); ng_va = gi_va.max() + 1
    yva = y[va].astype(np.float64)

    if obj == 'softmax':
        gi_tr = make_group_index(rid[tr])
        custom = race_softmax_obj(gi_tr, gi_tr.max() + 1, y[tr].astype(np.float64))

        def metric(preds, dm):
            pv = softmax_by_group(preds.astype(np.float64), gi_va, ng_va)
            return 'rll', float(-np.log(np.clip(pv[yva == 1], 1e-12, None)).mean())
        p.update(objective='reg:squarederror', base_score=0.0,
                 disable_default_eval_metric=1)
        b = xgb.train(p, dtr, nround, evals=[(dva, 'val')], obj=custom,
                      custom_metric=metric, early_stopping_rounds=esr,
                      maximize=False, verbose_eval=False)
        s = b.predict(dva, iteration_range=(0, b.best_iteration + 1)).astype(np.float64)
        pn = softmax_by_group(s, gi_va, ng_va)
    else:
        p.update(objective='binary:logistic', eval_metric='logloss')
        b = xgb.train(p, dtr, nround, evals=[(dva, 'val')],
                      early_stopping_rounds=esr, verbose_eval=False)
        s = b.predict(dva, iteration_range=(0, b.best_iteration + 1)).astype(np.float64)
        ser = pd.Series(s)
        pn = (s / ser.groupby(rid[va]).transform('sum').to_numpy())

    rll = float(-np.log(np.clip(pn[yva == 1], 1e-12, None)).mean())
    from sklearn.metrics import roc_auc_score
    return rll, float(roc_auc_score(yva, s)), b.best_iteration, b


def main():
    df, feats = load(IRE)
    X = df[feats].to_numpy(np.float32)
    y = df['win'].to_numpy(np.float32)
    rid = df['race_id'].to_numpy()
    tr, va, te, bounds = chrono_split(df)
    print(f'obj={OBJ} ire={IRE} split={bounds} rows={len(df)}', flush=True)

    # ---- stage 1: random search on the single chronological validation block ----
    trials = []
    for i in range(NTRIAL):
        hp = sample_params()
        t0 = time.time()
        rll, auc, bi, _ = fit_eval(X, y, rid, tr, va, hp)
        trials.append(dict(hp, rll=rll, auc=auc, best_iter=bi))
        print(f'[{i:02d}] rll={rll:.4f} auc={auc:.4f} it={bi:4d} ({time.time()-t0:.0f}s) {hp}',
              flush=True)
    tdf = pd.DataFrame(trials).sort_values('rll')
    tag = ('_ire' if IRE else '') + f'_{OBJ}'
    tdf.to_csv(HERE + f'/results_tune{tag}.csv', index=False)
    print('\nTOP 5 by val race-logloss:\n', tdf.head(5).to_string(index=False), flush=True)

    # ---- stage 2: walk-forward CV on the top 3 configurations ----
    folds = walk_forward_folds(df[~te.astype(bool)].reset_index(drop=True), n_folds=3,
                               train_frac=0.55)
    sub = df[~te.astype(bool)].reset_index(drop=True)
    Xs = sub[feats].to_numpy(np.float32); ys = sub['win'].to_numpy(np.float32)
    rids = sub['race_id'].to_numpy()
    keys = [c for c in tdf.columns if c not in ('rll', 'auc', 'best_iter')]
    wf = []
    for rank in range(min(2, len(tdf))):
        hp = {k: tdf.iloc[rank][k] for k in keys}
        hp['max_depth'] = int(hp['max_depth'])
        scores, iters = [], []
        for fi, (ftr, fva) in enumerate(folds):
            rll, auc, bi, _ = fit_eval(Xs, ys, rids, ftr, fva, hp)
            scores.append((rll, auc)); iters.append(bi)
            print(f'  cfg{rank} fold{fi}: rll={rll:.4f} auc={auc:.4f} it={bi}', flush=True)
        arr = np.array(scores)
        wf.append(dict(cfg=rank, mean_rll=arr[:, 0].mean(), mean_auc=arr[:, 1].mean(),
                       mean_iter=float(np.mean(iters)), **hp))
        print(f'cfg{rank}: WF mean rll={arr[:,0].mean():.4f} auc={arr[:,1].mean():.4f}',
              flush=True)
    wdf = pd.DataFrame(wf).sort_values('mean_rll')
    wdf.to_csv(HERE + f'/results_walkforward{tag}.csv', index=False)
    print('\nWALK-FORWARD:\n', wdf.to_string(index=False))
    best = wdf.iloc[0]
    out = {k: (int(best[k]) if k == 'max_depth' else float(best[k])) for k in keys}
    out['_mean_iter'] = float(best['mean_iter'])
    json.dump(out, open(HERE + f'/best_params{tag}.json', 'w'), indent=1)
    print('saved best params', out)


if __name__ == '__main__':
    main()
