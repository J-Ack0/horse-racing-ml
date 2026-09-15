"""Shared data loading, chronological splitting, objectives and metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import os
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')


def load(ire_only=False, shuffle_within_race=True):
    df = pd.read_pickle(f'{CACHE}/features.pkl')
    if shuffle_within_race:
        # The source table stores each race's rows in FINISHING order (58% of races have
        # the winner as their first row). Any evaluation that breaks tied scores by row
        # order would therefore silently read the result off the row index. Shuffle rows
        # once with a fixed seed, then stable-sort back by (date, race_id) so races stay
        # contiguous (needed for XGBoost group definitions) but within-race order is random.
        rs = np.random.default_rng(1234).permutation(len(df))
        df = df.iloc[rs].sort_values(['date', 'race_id'], kind='stable').reset_index(drop=True)
    if ire_only:
        df = df[df['is_ire'] == 1].reset_index(drop=True)
    feats = [l.strip() for l in open(f'{CACHE}/feature_names.txt') if l.strip()]
    feats = [f for f in feats if f in df.columns]
    if ire_only:
        feats = [f for f in feats if f != 'is_ire']
    return df, feats


def chrono_split(df, fracs=(0.64, 0.16, 0.20)):
    """Split by DATE so no race straddles a boundary; boundaries at row-count quantiles."""
    dates = df['date'].values
    order = np.sort(np.unique(dates))
    cum = df.groupby('date').size().sort_index().cumsum() / len(df)
    d1 = cum[cum >= fracs[0]].index[0]
    d2 = cum[cum >= fracs[0] + fracs[1]].index[0]
    tr = df['date'] < d1
    va = (df['date'] >= d1) & (df['date'] < d2)
    te = df['date'] >= d2
    return tr.values, va.values, te.values, (d1, d2)


def walk_forward_folds(df, n_folds=4, train_frac=0.5):
    """Rolling-origin folds: expanding train window, contiguous later validation block."""
    cum = (df.groupby('date').size().sort_index().cumsum() / len(df))
    folds = []
    span = (1.0 - train_frac) / n_folds
    for k in range(n_folds):
        a = train_frac + k * span
        b = a + span
        da = cum[cum >= a].index[0]
        db = cum[cum >= b].index[0] if (cum >= b).any() else None
        tr = (df['date'] < da).values
        va = ((df['date'] >= da) & ((df['date'] < db) if db is not None else True)).values
        folds.append((tr, va))
    return folds


# ---------------------------------------------------------------- metrics
def race_metrics(df_idx, y, p, pn=None, prefix=''):
    """Metrics. `p` = raw score (any monotone scale, used for AUC/AP/ranking).
    `pn` = per-race normalised probability used for calibration metrics; if None it is
    derived as p / sum(p) within race (valid only when p is a probability)."""
    d = pd.DataFrame({'race_id': df_idx['race_id'].values, 'y': y, 'p': p})
    out = {}
    out['auc'] = roc_auc_score(y, p)
    out['ap'] = average_precision_score(y, p)

    g = d.groupby('race_id', sort=False)
    d['pn'] = pn if pn is not None else d['p'] / g['p'].transform('sum')
    if p.min() >= 0 and p.max() <= 1:
        out['brier_raw'] = brier_score_loss(y, np.clip(p, 1e-7, 1 - 1e-7))
    else:
        out['brier_raw'] = np.nan
    out['brier_norm'] = np.mean((d['pn'] - d['y']) ** 2)
    eps = 1e-12
    out['race_logloss'] = -np.log(np.clip(d.loc[d['y'] == 1, 'pn'], eps, None)).mean()

    # ranking within race (rank 1 = highest p)
    d['rank'] = g['p'].rank(ascending=False, method='first')
    win_rank = d.loc[d['y'] == 1, 'rank']
    out['top1'] = float((win_rank == 1).mean())
    out['top3'] = float((win_rank <= 3).mean())
    out['mrr'] = float((1.0 / win_rank).mean())
    # NDCG@3 with binary gain: DCG = 1/log2(rank+1) if winner in top 3
    out['ndcg3'] = float(np.where(win_rank <= 3, 1.0 / np.log2(win_rank + 1), 0.0).mean())
    return {prefix + k: v for k, v in out.items()}


def ece(y, p, bins=15):
    """Expected calibration error (equal-count bins)."""
    q = pd.qcut(pd.Series(p).rank(method='first'), bins, labels=False)
    d = pd.DataFrame({'y': y, 'p': p, 'b': q})
    g = d.groupby('b')
    w = g.size() / len(d)
    return float((w * (g['p'].mean() - g['y'].mean()).abs()).sum())


# ---------------------------------------------------------------- softmax objective
def make_group_index(race_ids):
    """Map race_id array -> integer group index 0..G-1 (contiguous by first appearance)."""
    codes, _ = pd.factorize(race_ids, sort=False)
    return codes.astype(np.int64)


def softmax_by_group(scores, gidx, n_groups):
    """Numerically stable per-group softmax."""
    mx = np.full(n_groups, -np.inf)
    np.maximum.at(mx, gidx, scores)
    e = np.exp(scores - mx[gidx])
    s = np.zeros(n_groups)
    np.add.at(s, gidx, e)
    return e / s[gidx]


def race_softmax_obj(gidx, n_groups, y):
    """XGBoost custom objective: per-race softmax cross-entropy (conditional logit).

    L = -sum_races log p_winner ;  p_i = exp(s_i)/sum_{j in race} exp(s_j)
    grad_i = p_i * (#winners in race) - y_i        (usually #winners = 1)
    hess_i = p_i * (1 - p_i)    (diagonal approximation of the multinomial Hessian)
    """
    nw = np.zeros(n_groups)
    np.add.at(nw, gidx, y)

    def obj(preds, dtrain):
        p = softmax_by_group(preds, gidx, n_groups)
        grad = p * nw[gidx] - y
        hess = np.maximum(p * (1.0 - p), 1e-6)
        return grad.astype(np.float32), hess.astype(np.float32)
    return obj


def masked_softmax(scores, gidx, n_groups, mask):
    """Softmax over each group restricted to mask; entries outside mask get 0."""
    s = np.where(mask, scores, -np.inf)
    mx = np.full(n_groups, -np.inf)
    np.maximum.at(mx, gidx, s)
    e = np.where(mask, np.exp(s - mx[gidx]), 0.0)
    tot = np.zeros(n_groups)
    np.add.at(tot, gidx, e)
    tot = np.where(tot > 0, tot, 1.0)
    return e / tot[gidx]


def plackett_luce_topk_obj(gidx, n_groups, rank, K=3):
    """Plackett-Luce top-K partial likelihood as an XGBoost custom objective.

    L = -sum_races sum_{k=1..K} [ s_(k) - log sum_{j: rank_j >= k} exp(s_j) ]

    i.e. the winner is chosen from the whole field, the runner-up from the field minus
    the winner, and so on -- the standard sequential (Luce choice axiom) model of a race.
    Finishing ranks beyond first are POST-RACE, but they are used only as TRAINING
    LABELS, never as features, which is legitimate.
    """
    rank = np.asarray(rank)
    masks = [rank >= k for k in range(1, K + 1)]
    chosen = [rank == k for k in range(1, K + 1)]

    def obj(preds, dtrain):
        s = preds.astype(np.float64)
        grad = np.zeros_like(s)
        hess = np.zeros_like(s)
        for m, c in zip(masks, chosen):
            p = masked_softmax(s, gidx, n_groups, m)
            grad += p - c.astype(np.float64)
            hess += p * (1.0 - p)
        return grad.astype(np.float32), np.maximum(hess, 1e-6).astype(np.float32)
    return obj
