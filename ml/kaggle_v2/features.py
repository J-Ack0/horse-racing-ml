"""
Leak-free feature engineering for UK+IRE horse racing win prediction.

Design rules (see project brief):
  * NEVER use current-row rpr / ts / sp / pos / btn / ovr_btn / time / prize / comment.
      - rpr, ts  : post-race performance figures  (leak)
      - sp       : market odds                    (leak + explicitly excluded)
      - prize    : money won by THIS horse in THIS race (leak) -- found in this pass
  * `or` (Official Rating) IS pre-race and is used.
  * Every historical statistic is computed strictly point-in-time at *day* granularity:
      cumulative totals up to but EXCLUDING the current race date. This is stricter than
      a plain cumsum-minus-self, which would leak results of earlier races on the same day
      (and, for trainers/sires with several runners in one race, results from the very
      same race).
  * Raw horse / jockey / trainer / sire / dam names are NEVER emitted as features -- they
    are only group keys for aggregation.

Outputs a parquet cache of the engineered matrix.
"""
import os
import sqlite3
import numpy as np
import pandas as pd

DB = 'data_ext/raceform.db'
CACHE = 'ml/kaggle_v2/cache'

# Model window (matches the session baseline: last 3 years of data, data ends 2026-05-27)
MODEL_START = '2023-05-27'
# Extra history loaded purely to warm up the point-in-time rolling statistics.
HISTORY_START = os.environ.get('HISTORY_START', '2021-01-01')

GOING_MAP = {
    'Hard': 0, 'Firm': 1, 'Good To Firm': 2, 'Good': 3, 'Good To Yielding': 3.5,
    'Yielding': 4, 'Good To Soft': 4, 'Yielding To Soft': 4.5, 'Soft': 5,
    'Soft To Heavy': 5.5, 'Heavy': 6,
    # all-weather surfaces
    'Standard': 3, 'Standard To Slow': 3.5, 'Slow': 4, 'Standard To Fast': 2.5,
    'Fast': 2, 'Sloppy': 4.5, 'Muddy': 5, 'Holding': 5.5,
}


def wilson(wins, n, z=1.96):
    """Wilson lower bound of a binomial proportion. Shrinks small samples toward 0."""
    n = np.asarray(n, dtype=float)
    wins = np.asarray(wins, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        p = np.where(n > 0, wins / np.maximum(n, 1), 0.0)
        denom = 1 + z * z / n
        centre = p + z * z / (2 * n)
        adj = z * np.sqrt(np.maximum(p * (1 - p) / n + z * z / (4 * n * n), 0))
        out = (centre - adj) / denom
    return np.where(n > 0, out, 0.0)


def parse_dist(d):
    """'2m5½f' / '7f' / '1m' -> furlongs (float)."""
    if not isinstance(d, str) or not d.strip():
        return np.nan
    s = d.strip().replace('½', '.5').replace('¼', '.25').replace('¾', '.75')
    f = 0.0
    try:
        if 'm' in s:
            miles, _, rest = s.partition('m')
            f += float(miles) * 8 if miles else 0.0
            s = rest
        if 'f' in s:
            fur, _, _ = s.partition('f')
            f += float(fur) if fur else 0.0
        return f if f > 0 else np.nan
    except ValueError:
        return np.nan


def parse_wgt(w):
    """'11-0' stones-pounds -> lbs."""
    if not isinstance(w, str) or '-' not in w:
        return np.nan
    st, _, lb = w.partition('-')
    try:
        return float(st) * 14 + float(lb)
    except ValueError:
        return np.nan


def to_num(s):
    return pd.to_numeric(s.astype(str).str.strip().replace(
        {'': None, '-': None, '–': None, '—': None}), errors='coerce')


def load_raw():
    con = sqlite3.connect(DB)
    cols = ("date, course, race_id, off, race_name, type, class, pattern, age_band, "
            "sex_rest, dist, going, ran, num, pos, draw, horse, age, sex, wgt, hg, "
            "jockey, trainer, [or], rpr, ts, sire, dam, damsire")
    q = (f"SELECT {cols} FROM data WHERE date != 'date' AND date >= '{HISTORY_START}' "
         f"AND (course NOT LIKE '%(%' OR course LIKE '%(IRE)' OR course LIKE '%(AW)')")
    df = pd.read_sql(q, con)
    con.close()
    return df


def build(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()]

    # ---------------- target ----------------
    pos_num = to_num(df['pos'])
    df['win'] = (df['pos'].astype(str).str.strip() == '1').astype(np.int8)
    df['finished'] = pos_num.notna().astype(np.int8)
    df['pos_num'] = pos_num

    df['ran'] = to_num(df['ran'])
    df = df[(df['ran'] >= 3) & (df['ran'].notna())]

    # Drop races with a result but zero winners (data errors). A race where
    # EVERY row is unresolved (pos not yet known - i.e. today's/a future
    # racecard, used at inference time) is deliberately kept: it hasn't run
    # yet, so "zero winners so far" doesn't mean the data is broken.
    wins_per_race = df.groupby('race_id')['win'].transform('sum')
    any_result_per_race = df.groupby('race_id')['finished'].transform('max')
    df = df[(wins_per_race >= 1) | (any_result_per_race == 0)]

    df = df.sort_values(['date', 'race_id']).reset_index(drop=True)

    # ---------------- static pre-race ----------------
    df['is_ire'] = df['course'].str.contains(r'\(IRE\)', regex=True).astype(np.int8)
    df['is_aw'] = df['course'].str.contains(r'\(AW\)', regex=True).astype(np.int8)
    df['field_size'] = df['ran']
    df['draw'] = to_num(df['draw'])
    df['has_draw'] = df['draw'].notna().astype(np.int8)
    df['draw_pct'] = df['draw'] / df['ran']
    df['num'] = to_num(df['num'])
    df['age'] = to_num(df['age'])
    df['wgt_lbs'] = df['wgt'].map(parse_wgt)
    df['dist_f'] = df['dist'].map(parse_dist)
    df['log_dist'] = np.log1p(df['dist_f'])
    df['going_num'] = df['going'].str.strip().map(GOING_MAP)
    df['class_num'] = to_num(df['class'].str.extract(r'(\d)', expand=False))
    pat = df['pattern'].fillna('').str.strip()
    df['is_group'] = pat.str.contains('Group|Grade', case=False).astype(np.int8)
    df['pattern_lvl'] = to_num(pat.str.extract(r'([123])', expand=False)).fillna(0)
    rn = df['race_name'].fillna('')
    df['is_handicap'] = rn.str.contains('Handicap|H\'cap', case=False, regex=True).astype(np.int8)
    df['is_maiden'] = rn.str.contains('Maiden', case=False).astype(np.int8)
    df['is_novice'] = rn.str.contains('Novice|Nov ', case=False, regex=True).astype(np.int8)
    df['is_seller'] = rn.str.contains('Seller|Claim', case=False, regex=True).astype(np.int8)
    rtype = df['type'].fillna('').str.strip()
    for t, name in [('Flat', 'flat'), ('Hurdle', 'hurdle'), ('Chase', 'chase'), ('NH Flat', 'nhflat')]:
        df[f'type_{name}'] = (rtype == t).astype(np.int8)
    df['is_jumps'] = (df['type_hurdle'] | df['type_chase'] | df['type_nhflat']).astype(np.int8)
    sx = df['sex'].fillna('').str.strip().str.upper()
    for s in ['G', 'C', 'F', 'M', 'H']:
        df[f'sex_{s}'] = (sx == s).astype(np.int8)
    hg = df['hg'].fillna('').str.strip()
    df['has_hg'] = (hg != '').astype(np.int8)
    df['month'] = df['date'].dt.month
    df['dow'] = df['date'].dt.dayofweek
    df['off_hour'] = to_num(df['off'].astype(str).str.extract(r'^(\d+)', expand=False))

    df['or'] = to_num(df['or'])
    df['or_missing'] = df['or'].isna().astype(np.int8)
    df['rpr_raw'] = to_num(df['rpr'])   # ONLY used to build lagged prior_* below
    df['ts_raw'] = to_num(df['ts'])

    # ---------------- per-horse lagged history (strict shift) ----------------
    g = df.groupby('horse', sort=False)
    df['h_runs_prior'] = g.cumcount()
    df['h_wins_prior'] = g['win'].cumsum() - df['win']
    df['h_fin_prior'] = g['finished'].cumsum() - df['finished']
    # normalised finishing position of prior runs (1 = won, 0 = last)
    rel = np.where(df['ran'] > 1, 1.0 - (df['pos_num'] - 1) / (df['ran'] - 1), np.nan)
    df['_rel'] = np.where(df['finished'] == 1, rel, 0.0)
    df['h_rel_sum_prior'] = g['_rel'].cumsum() - df['_rel']
    df['h_rel_mean_prior'] = df['h_rel_sum_prior'] / df['h_runs_prior'].replace(0, np.nan)
    df['h_win_rate'] = wilson(df['h_wins_prior'], df['h_runs_prior'])
    df['h_dnf_rate'] = np.where(df['h_runs_prior'] > 0,
                                1 - df['h_fin_prior'] / df['h_runs_prior'].replace(0, np.nan), 0.0)

    # EMA of prior relative finish (shift so current run excluded)
    df['h_rel_ema'] = (g['_rel'].transform(lambda s: s.ewm(alpha=0.4, adjust=False).mean())
                       .groupby(df['horse']).shift(1))
    df['h_win_ema'] = (g['win'].transform(lambda s: s.ewm(alpha=0.3, adjust=False).mean())
                       .groupby(df['horse']).shift(1))

    # lagged ratings -- legitimate: this horse's figure from ITS OWN previous run
    for src, name in [('rpr_raw', 'prior_rpr'), ('ts_raw', 'prior_ts'), ('or', 'prior_or')]:
        df[name] = g[src].shift(1)
    df[ 'prior_rpr_best3'] = (g['rpr_raw'].transform(lambda s: s.shift(1).rolling(3, min_periods=1).max()))
    df['prior_ts_best3'] = (g['ts_raw'].transform(lambda s: s.shift(1).rolling(3, min_periods=1).max()))
    df['prior_rpr_best_career'] = g['rpr_raw'].transform(lambda s: s.shift(1).cummax())
    df['or_change'] = df['or'] - df['prior_or']
    df['prior_rpr_missing'] = df['prior_rpr'].isna().astype(np.int8)
    df['prior_ts_missing'] = df['prior_ts'].isna().astype(np.int8)

    # layoff
    df['last_date'] = g['date'].shift(1)
    df['days_since_run'] = (df['date'] - df['last_date']).dt.days
    df['log_days_since'] = np.log1p(df['days_since_run'])
    df['first_run'] = (df['h_runs_prior'] == 0).astype(np.int8)

    # headgear change vs previous run
    df['_hg'] = hg
    df['prev_hg'] = g['_hg'].shift(1)
    df['hg_change'] = ((df['_hg'] != df['prev_hg']) & df['h_runs_prior'].gt(0)).astype(np.int8)
    df['hg_first_time'] = ((df['has_hg'] == 1) & (df['prev_hg'].fillna('') == '')).astype(np.int8)

    # horse x context prior experience (course / distance band / going band / type)
    df['dist_band'] = pd.cut(df['dist_f'], [0, 6.5, 8.5, 11, 14, 20, 100], labels=False)
    df['going_band'] = pd.cut(df['going_num'], [-1, 2.5, 3.5, 4.5, 10], labels=False)
    for key, tag in [(['horse', 'course'], 'hc'), (['horse', 'dist_band'], 'hd'),
                     (['horse', 'going_band'], 'hg2'), (['horse', 'type'], 'ht')]:
        gg = df.groupby(key, sort=False, dropna=False)
        df[f'{tag}_runs'] = gg.cumcount()
        df[f'{tag}_wins'] = gg['win'].cumsum() - df['win']
        df[f'{tag}_wr'] = wilson(df[f'{tag}_wins'], df[f'{tag}_runs'])

    # ---------------- entity stats: strict day-boundary cumulative ----------------
    def day_stats(keys, tag, extra_hit=None):
        """Cumulative runs/wins for `keys` up to but EXCLUDING the current date."""
        agg = df.groupby(keys + ['date'], sort=True, dropna=False).agg(
            n=('win', 'size'), w=('win', 'sum'))
        agg = agg.sort_index()
        gk = agg.groupby(level=list(range(len(keys))))
        agg[f'{tag}_runs'] = gk['n'].cumsum() - agg['n']
        agg[f'{tag}_wins'] = gk['w'].cumsum() - agg['w']
        out = agg[[f'{tag}_runs', f'{tag}_wins']].reset_index()
        merged = df[keys + ['date']].merge(out, on=keys + ['date'], how='left')
        df[f'{tag}_runs'] = merged[f'{tag}_runs'].values
        df[f'{tag}_wins'] = merged[f'{tag}_wins'].values
        df[f'{tag}_wr'] = wilson(df[f'{tag}_wins'], df[f'{tag}_runs'])
        df[f'{tag}_logn'] = np.log1p(df[f'{tag}_runs'])

    day_stats(['jockey'], 'jky')
    day_stats(['trainer'], 'trn')
    day_stats(['jockey', 'trainer'], 'jt')
    day_stats(['trainer', 'course'], 'tc')
    day_stats(['jockey', 'course'], 'jc')
    day_stats(['sire'], 'sire')
    day_stats(['dam'], 'dam')
    day_stats(['damsire'], 'dsire')
    day_stats(['trainer', 'type'], 'tt')

    # The per-column assignments above (and the horse/course/dist/going/type
    # loop before them) leave the frame heavily fragmented -- a single
    # defragmenting copy here is a pure performance fix (same values, better
    # memory layout), not a behavior change.
    df = df.copy()

    # ---------------- race-relative (within-race) transforms ----------------
    # These use only OTHER runners' PRE-RACE features -> not leakage.
    rel_src = ['or', 'prior_rpr', 'prior_ts', 'h_win_rate', 'jky_wr', 'trn_wr',
               'h_rel_ema', 'wgt_lbs', 'prior_rpr_best3', 'jt_wr', 'age',
               'h_runs_prior', 'days_since_run', 'sire_wr']
    grp = df.groupby('race_id', sort=False)
    for c in rel_src:
        m = grp[c].transform('mean')
        s = grp[c].transform('std')
        df[f'{c}_z'] = (df[c] - m) / s.replace(0, np.nan)
        df[f'{c}_rk'] = grp[c].rank(pct=True, na_option='keep')
    df['or_max_race'] = grp['or'].transform('max')
    df['or_gap_top'] = df['or'] - df['or_max_race']
    df['n_with_or'] = grp['or'].transform('count')

    return df


FEATURES = None


def feature_list(df):
    drop = {'date', 'course', 'race_id', 'off', 'race_name', 'type', 'class', 'pattern',
            'age_band', 'sex_rest', 'dist', 'going', 'pos', 'horse', 'sex', 'wgt', 'hg',
            'jockey', 'trainer', 'sire', 'dam', 'damsire', 'rpr', 'ts', 'win', 'ran',
            'pos_num', 'finished', '_rel', 'rpr_raw', 'ts_raw', 'last_date', '_hg',
            'prev_hg', 'h_rel_sum_prior', 'or_max_race', 'dist_band', 'going_band'}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def main():
    os.makedirs(CACHE, exist_ok=True)
    print('loading...', flush=True)
    raw = load_raw()
    print('raw rows', len(raw), flush=True)
    df = build(raw)
    del raw
    df = df[df['date'] >= MODEL_START].reset_index(drop=True)
    feats = feature_list(df)
    # pos_num/finished are LABELS (post-race) kept for ranking objectives -- never features.
    keep = ['date', 'race_id', 'win', 'course', 'pos_num', 'finished'] + \
           [c for c in feats if c != 'course']
    keep = list(dict.fromkeys(keep))
    out = df[keep].copy()
    for c in feats:
        out[c] = out[c].astype(np.float32)
    out.to_pickle(os.environ.get('OUT_PKL', f'{CACHE}/features.pkl'))
    with open(f'{CACHE}/feature_names.txt', 'w') as f:
        f.write('\n'.join(feats))
    print('rows', len(out), 'features', len(feats))
    print('date range', out['date'].min(), out['date'].max())
    print('win rate', out['win'].mean().round(4), 'races', out['race_id'].nunique())
    print('IRE rows', int(out['is_ire'].to_numpy().sum()))


if __name__ == '__main__':
    main()
