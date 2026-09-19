#!/usr/bin/env bash
# Daily live pipeline driver, called by the racingapi-daily-* systemd user units.
#   run_daily.sh fetch   racecard for today, then inference on it (needs ~3 GB free RAM)
#   run_daily.sh score   score today's predictions against /results/today/free
# The Free plan only serves today's results, so `score` must run on the same day.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/venv/bin/python"
DAY="$(date +%F)"

case "${1:-}" in
  fetch)
    "$PY" data_collection/fetch_daily_racecards.py --day today
    "$PY" ml/kaggle_v2/inference.py --date "$DAY"
    ;;
  score)
    "$PY" ml/kaggle_v2/score_predictions.py --date "$DAY"
    ;;
  *)
    echo "usage: run_daily.sh fetch|score" >&2; exit 2 ;;
esac
