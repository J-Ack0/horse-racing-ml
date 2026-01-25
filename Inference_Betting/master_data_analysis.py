# audit_master.py
"""
Audit script: loop Inference_Betting/*.csv and Inference_Actuals/*, build master.parquet
Outputs:
  - output/master.parquet
  - output/aggregates/roi_by_track.csv
  - output/aggregates/roi_by_odds_bucket.csv
  - output/aggregates/roi_by_edge_bucket.csv
  - output/aggregates/ev_deciles.csv
Usage:
  python audit_master.py --bet-dir Inference_Betting --actual-dir Inference_Actuals --out output
"""
import re
import os
import glob
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------------- utilities ----------------
def normalize_horse(name):
    if pd.isna(name):
        return ''
    s = str(name).lower()
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def parse_date_from_filename(fn):
    # find YYYY-MM-DD or YYYY_MM_DD or YYYYMMDD
    m = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', fn)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None

def safe_num(x):
    try:
        if pd.isna(x) or x=='':
            return np.nan
        return float(str(x).strip())
    except Exception:
        return np.nan

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ---------------- core logic ----------------
def read_csv_best(path):
    # robust wrapper
    return pd.read_csv(path, dtype=str).fillna('')

def build_race_key_from_row(row, file_date):
    # prefer Race_id if present
    if 'Race_id' in row and row['Race_id']:
        return str(row['Race_id'])
    track = (row.get('track_name') or row.get('track') or '').strip()
    time = (row.get('time') or row.get('race_time') or '').strip()
    # include file date if present
    date = file_date or ''
    return f"{track}__{time}__{date}"

def load_actuals_map(actual_dir):
    # Read all actual files into map by date-string (yyy-mm-dd) if possible, else by filename
    actual_files = glob.glob(os.path.join(actual_dir, '*'))
    actual_map = {}
    for f in actual_files:
        key = parse_date_from_filename(os.path.basename(f)) or os.path.basename(f)
        try:
            df = read_csv_best(f)
            df['_source_path'] = f
            actual_map[key] = df
        except Exception as e:
            print("warn: failed to read actuals", f, e)
    return actual_map

def main(bet_dir, actual_dir, out_dir):
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, 'aggregates'))

    actual_map = load_actuals_map(actual_dir)
    bet_files = sorted(glob.glob(os.path.join(bet_dir, '*.csv')))

    master_rows = []

    print(f"Found {len(bet_files)} bet files, {len(actual_map)} actual-date groups")

    for bf in tqdm(bet_files, desc="Bet files"):
        bname = os.path.basename(bf)
        bet_date = parse_date_from_filename(bname)
        try:
            bets = read_csv_best(bf)
        except Exception as e:
            print("skip unreadable bet file", bf, e)
            continue

        # choose matching actuals: prioritize exact date match, else fallback to any file whose name contains date parts
        matched_actual_df = None
        if bet_date:
            # actual map keys might be yyyy-mm-dd or yyyy_mm_dd; try multiple forms
            keys_to_try = [bet_date, bet_date.replace('-', '_'), bet_date.replace('-', '')]
            for k in keys_to_try:
                if k in actual_map:
                    matched_actual_df = actual_map[k]
                    break
        if matched_actual_df is None:
            # try fuzzy: any actual file with same year-month-day tokens
            for k, df in actual_map.items():
                if bet_date and all(tok in k for tok in bet_date.split('-')):
                    matched_actual_df = df
                    break
        # If still none, we'll proceed — rows can be unmatched; composite keys will be used
        # Turn both frames into normalized rows
        bets_rows = bets.to_dict(orient='records')
        actuals_rows = (matched_actual_df.to_dict(orient='records') if matched_actual_df is not None else [])

        # index actuals by race_key + horse normalized
        actual_index = {}
        for r in actuals_rows:
            if matched_actual_df is not None and '_source_path' in matched_actual_df.columns:
                actual_path = matched_actual_df['_source_path'].iloc[0]
                file_date = parse_date_from_filename(os.path.basename(actual_path)) or bet_date
            else:
                file_date = bet_date
            race_key = build_race_key_from_row(r, file_date)
            horse_key = normalize_horse(r.get('horse_name_norm') or r.get('horse_name') or '')
            actual_index[(race_key, horse_key)] = r

        # produce master rows by matching bet row -> actual row
        for br in bets_rows:
            file_date = bet_date
            race_key = build_race_key_from_row(br, file_date)
            horse_key = normalize_horse(br.get('horse_name_odds') or br.get('horse_name') or '')
            actual_row = actual_index.get((race_key, horse_key), None)
            merged = {}
            # add all bet fields (prefix bet_ to avoid collision)
            for k,v in br.items():
                merged[f"bet_{k}"] = v
            # attach actuals if present (without prefix)
            if actual_row:
                for k,v in actual_row.items():
                    merged[k] = v
            # canonical fields
            merged['race_date'] = file_date
            merged['race_key'] = race_key
            merged['horse_key'] = horse_key
            # numeric conversions
            merged['kelly_stake'] = safe_num(merged.get('bet_kelly_stake') or merged.get('kelly_stake') or '')
            merged['decimal_odds'] = safe_num(merged.get('decimal_odds') or merged.get('bet_decimal_odds') or merged.get('odds_live') or '')
            merged['edge'] = safe_num(merged.get('edge') or merged.get('raw_edge') or merged.get('bet_edge') or '')
            merged['ev'] = safe_num(merged.get('ev') or merged.get('bet_ev') or '')
            # actual position
            ap = merged.get('actual_position') or merged.get('inferred_position') or merged.get('bet_actual_position') or ''
            merged['actual_position'] = int(ap) if str(ap).strip().isdigit() else np.nan
            master_rows.append(merged)

    if len(master_rows) == 0:
        print("No master rows produced. Exiting.")
        return

    master_df = pd.DataFrame(master_rows)

    # standardized columns / fill NaNs
    master_df['kelly_stake'] = pd.to_numeric(master_df['kelly_stake'], errors='coerce').fillna(0.0)
    master_df['decimal_odds'] = pd.to_numeric(master_df['decimal_odds'], errors='coerce')
    master_df['edge'] = pd.to_numeric(master_df['edge'], errors='coerce')
    master_df['ev'] = pd.to_numeric(master_df['ev'], errors='coerce')

    # derived: stake (we consider kelly_stake as the stake), returned, profit
    master_df['stake'] = master_df['kelly_stake']
    master_df['returned'] = master_df.apply(lambda r: r['stake'] * r['decimal_odds'] if not np.isnan(r['decimal_odds']) and r['actual_position']=="1" else 0.0, axis=1)
    master_df['profit'] = master_df['returned'] - master_df['stake']

    # write master.parquet
    out_master = os.path.join(out_dir, 'master.parquet')
    master_df.to_parquet(out_master, index=False)
    print("Wrote master:", out_master)

    # Aggregates
    agg_dir = os.path.join(out_dir, 'aggregates')
    ensure_dir(agg_dir)

    # ROI by track
    if 'track_name' in master_df.columns:
        master_df['track_name'] = master_df['track_name']
    elif 'bet_track_name' in master_df.columns:
        master_df['track_name'] = master_df['bet_track_name']
    elif 'bet_track' in master_df.columns:
        master_df['track_name'] = master_df['bet_track']
    else:
        master_df['track_name'] = ''
        master_df['track_name'] = master_df['track_name'].fillna('').astype(str)
    by_track = master_df.groupby('track_name').agg(
        bets=('stake', lambda s: (s>0).sum()),
        staked=('stake', 'sum'),
        returned=('returned', 'sum'),
        profit=('profit', 'sum')
    ).reset_index()
    by_track['roi'] = by_track.apply(lambda r: r['profit'] / r['staked'] if r['staked']>0 else np.nan, axis=1)
    by_track.to_csv(os.path.join(agg_dir, 'roi_by_track.csv'), index=False)

    # odds buckets (ex: 0-2,2-5,5-10,10-20,20-50,50+)
    bins = [0,2,5,10,20,50,999999]
    labels = ['0-2','2-5','5-10','10-20','20-50','50+']
    master_df['odds_bucket'] = pd.cut(master_df['decimal_odds'].fillna(999999), bins=bins, labels=labels, include_lowest=True)
    by_odds = master_df.groupby('odds_bucket').agg(bets=('stake', lambda s:(s>0).sum()), staked=('stake','sum'), returned=('returned','sum'), profit=('profit','sum')).reset_index()
    by_odds['roi'] = by_odds.apply(lambda r: r['profit'] / r['staked'] if r['staked']>0 else np.nan, axis=1)
    by_odds.to_csv(os.path.join(agg_dir, 'roi_by_odds_bucket.csv'), index=False)

    # edge buckets
    edge_bins = [-999,0,0.01,0.03,0.05,0.1,1]
    edge_labels = ['neg','0-0.01','0.01-0.03','0.03-0.05','0.05-0.1','0.1+']
    master_df['edge_bucket'] = pd.cut(master_df['edge'].fillna(-999), bins=edge_bins, labels=edge_labels)
    by_edge = master_df.groupby('edge_bucket').agg(bets=('stake', lambda s:(s>0).sum()), staked=('stake','sum'), returned=('returned','sum'), profit=('profit','sum')).reset_index()
    by_edge['roi'] = by_edge.apply(lambda r: r['profit'] / r['staked'] if r['staked']>0 else np.nan, axis=1)
    by_edge.to_csv(os.path.join(agg_dir, 'roi_by_edge_bucket.csv'), index=False)

    # EV deciles
    master_df['ev_filled'] = master_df['ev'].fillna(0.0)
    master_df['ev_decile'] = pd.qcut(master_df['ev_filled'].rank(method='first'), 10, labels=[f"decile_{i+1}" for i in range(10)])
    by_ev = master_df.groupby('ev_decile').agg(bets=('stake', lambda s:(s>0).sum()), staked=('stake','sum'), returned=('returned','sum'), profit=('profit','sum')).reset_index()
    by_ev['roi'] = by_ev.apply(lambda r: r['profit'] / r['staked'] if r['staked']>0 else np.nan, axis=1)
    by_ev.to_csv(os.path.join(agg_dir, 'ev_deciles.csv'), index=False)

    print("Aggregates written to", agg_dir)
    print("Done.")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bet-dir', default='Inference_Betting', help='Directory of bets CSVs')
    p.add_argument('--actual-dir', default='Inference_Actuals', help='Directory of actuals CSVs')
    p.add_argument('--out', default='output', help='Output directory')
    args = p.parse_args()
    main(args.bet_dir, args.actual_dir, args.out)
