import pandas as pd, glob, os
from common import HERE
files = ['results_objectives.csv', 'results_rank_retry.csv', 'results_ablation_nowarmup.csv',
         'results_final.csv', 'results_final_ire.csv']
cols = ['model', 'auc', 'ap', 'top1', 'top3', 'mrr', 'ndcg3', 'race_logloss', 'brier_norm', 'ece']
for f in files:
    p = os.path.join(HERE, f)
    if not os.path.exists(p):
        print('missing', f); continue
    d = pd.read_csv(p)
    print(f'\n### {f}')
    print(d[[c for c in cols if c in d.columns]].round(4).to_string(index=False))
