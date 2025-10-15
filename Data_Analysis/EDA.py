#!/usr/bin/env python3
"""
track_rating_analysis.py

Analyze whether 'rating' contributes to wins by track.

Usage:
    python track_rating_analysis.py --input path/to/your.csv --output track_summary.csv
"""

import argparse
import pandas as pd
import sys

def load_and_clean(path):
    df = pd.read_csv(path)
    # Coerce non-numeric ratings/positions to NaN
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['race_position'] = pd.to_numeric(df['race_position'], errors='coerce')
    # Drop any rows lacking those critical fields
    before = len(df)
    df = df.dropna(subset=['rating', 'race_position'])
    after = len(df)
    if after < before:
        print(f"[INFO] Dropped {before-after} rows due to non-numeric rating/position", file=sys.stderr)
    return df

def compute_overall_corr(df):
    df['win'] = (df['race_position'] == 1).astype(int)
    corr = df['rating'].corr(df['win'])
    return corr

def compute_track_summary(df):
    records = []
    for track, g in df.groupby('track_name'):
        # require at least one win and one non-win to get a valid correlation
        if len(g) < 2 or g['win'].nunique() < 2:
            corr = float('nan')
        else:
            corr = g['rating'].corr(g['win'])
        rec = {
            'track_name': track,
            'race_count': len(g),
            'win_rate': g['win'].mean(),
            'avg_rating_win': g.loc[g['win']==1, 'rating'].mean(),
            'avg_rating_nonwin': g.loc[g['win']==0, 'rating'].mean(),
            'rating_diff': g.loc[g['win']==1, 'rating'].mean() - g.loc[g['win']==0, 'rating'].mean(),
            'rating_win_corr': corr
        }
        records.append(rec)
    summary = pd.DataFrame(records)
    return summary

def main():
    p = argparse.ArgumentParser(description="Analyze rating vs wins by track")
    p.add_argument('--input', '-i', required=True, help="Path to CSV file")
    p.add_argument('--output', '-o', default='track_summary.csv', help="Path to write summary CSV")
    args = p.parse_args()

    df = load_and_clean(args.input)
    df['win'] = (df['race_position'] == 1).astype(int)

    overall_corr = compute_overall_corr(df)
    print(f"Overall Pearson correlation between rating and win-flag: {overall_corr:.3f}")

    summary = compute_track_summary(df)
    # Sort by race_count descending so busiest tracks appear first
    summary = summary.sort_values('race_count', ascending=False)

    # Print top few rows
    print("\nPer-track summary (top 10 by number of races):")
    print(summary.head(10).to_string(index=False, float_format='%.3f'))

    # Write full table
    summary.to_csv(args.output, index=False)
    print(f"\n[INFO] Full summary written to {args.output}")

if __name__ == '__main__':
    main()


