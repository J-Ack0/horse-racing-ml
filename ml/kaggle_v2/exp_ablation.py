"""Build the cold-start cache first:
    HISTORY_START=2023-05-27 OUT_PKL=$PWD/ml/kaggle_v2/cache/features_nowarm.pkl \
        python ml/kaggle_v2/features.py

Ablation: how much of the gain over tonight's 0.695 baseline comes from the extra
warm-up history (rolling stats primed on 2021-2023 data) versus the richer feature set?

Trains the SAME plain binary model with the SAME inherited hyperparameters on a feature
matrix built with HISTORY_START == MODEL_START (i.e. rolling stats start cold on
2023-05-27, exactly like the original 3-year-window baseline).
"""
import os, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from common import HERE, chrono_split, race_metrics, make_group_index, softmax_by_group

PKL = HERE + '/cache/features_nowarm.pkl'
df = pd.read_pickle(PKL)
rs = np.random.default_rng(1234).permutation(len(df))
df = df.iloc[rs].sort_values(['date', 'race_id'], kind='stable').reset_index(drop=True)
feats = [l.strip() for l in open(HERE + '/cache/feature_names.txt') if l.strip()]
feats = [f for f in feats if f in df.columns]
X = df[feats].to_numpy(np.float32); y = df['win'].to_numpy(np.float32)
rid = df['race_id'].to_numpy()
tr, va, te, b = chrono_split(df)
print('no-warm-up cache:', len(df), 'rows', len(feats), 'feats, split', b, flush=True)

p = dict(objective='binary:logistic', eval_metric='auc', max_depth=5, learning_rate=0.01,
         subsample=0.8, colsample_bytree=0.8, min_child_weight=1, nthread=4, seed=7)
dtr = xgb.DMatrix(X[tr], label=y[tr], missing=np.nan)
dva = xgb.DMatrix(X[va], label=y[va], missing=np.nan)
dte = xgb.DMatrix(X[te], missing=np.nan)
bst = xgb.train(p, dtr, 3000, evals=[(dva, 'val')], early_stopping_rounds=50,
                verbose_eval=False)
s = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1)).astype(np.float64)
pn = s / pd.Series(s).groupby(rid[te]).transform('sum').to_numpy()
m = race_metrics(df[te], y[te], s, pn)
print('best_iter', bst.best_iteration)
print({k: round(v, 4) for k, v in m.items()})
pd.DataFrame([dict(m, model='binary median/NaN, NO warm-up history')]).to_csv(
    HERE + '/results_ablation_nowarmup.csv', index=False)
