# Horse Racing ML

UK and Ireland horse-racing win prediction: a leak-free feature pipeline, race-conditional
XGBoost models (`ml/kaggle_v2/`), live racecards from The Racing API, and daily
inference and scoring. Built and run on a Raspberry Pi 5 (8 GB, no swap).

**Everything else is in [docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md)**: data window and
coverage, model results, the live pipeline, inference, scoring log, analysis, tests,
the legacy v1 system and its scraper audit, open TODOs, and how to re-verify each claim.

## Status (2026-09-18)

- Data window: 2015-01-01 to 2026-05-27 (`data_ext/raceform.db`); the gap to today
  needs a Standard Racing API plan to backfill.
- Model: UK+IRE `blend_all`, AUC 0.7445 on the held-out test split.
- Live scoring over 3 days (115 races): the model's top pick wins about 22.6% of races.

## Quick start

```bash
python3 -m venv venv
venv/bin/pip install requests python-dotenv pytest responses xgboost pandas numpy scikit-learn
# put USERNAME= and PASSWORD= (Racing API) in a repo-root .env (gitignored)

venv/bin/python data_collection/fetch_daily_racecards.py --day today
venv/bin/python ml/kaggle_v2/inference.py --date YYYY-MM-DD
venv/bin/python ml/kaggle_v2/score_predictions.py --date YYYY-MM-DD   # same day only

venv/bin/python -m pytest data_collection/tests -q
venv/bin/python -m pytest ml/kaggle_v2/tests -q
```

The static datasets live in `data_ext/` (untracked); see `data_ext/Kaggle_ReadMe.md` for
their provenance. The v1 Racing Post scrapers in `data_collection/scrape_*.py` are broken
and superseded by the API.

## Disclaimer

Built for research and learning. Betting carries real financial risk. Past model
performance does not predict future results.
