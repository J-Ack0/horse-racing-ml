#!/usr/bin/env bash
# Daily live pipeline driver, called by the racingapi-daily-* systemd user units.
#   run_daily.sh fetch   backfill yesterday's results, fetch today's racecard, fast inference (~1 min)
#   run_daily.sh score   score today's predictions against /results/today/free
# The Free plan only serves today's results, so `score` must run on the same day.
# Each run pushes a phone note via notify-phone.sh ([Done] on success, [Error] on
# failure). A notify failure or an offline phone (exit 2, queued) never fails the job.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/venv/bin/python"
DAY="$(date +%F)"
NOTIFY="$HOME/.local/bin/notify-phone.sh"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

notify() { "$NOTIFY" "$@" || true; }

MODE="${1:-}"
on_error() {
  local rc=$?
  notify error "Daily $MODE failed for $DAY (exit $rc)" "$(tail -n 15 "$LOG" 2>/dev/null)"
  exit "$rc"
}
trap on_error ERR

case "$MODE" in
  fetch)
    # 1. yesterday's settled results into the history (Standard plan). Not fatal: if it fails
    #    the day is still predicted, just without the newest results.
    if ! "$PY" data_collection/backfill_history.py 2>&1 | tee "$LOG"; then
      notify error "Results backfill failed for $DAY (predicting without the newest results)" "$(tail -n 8 "$LOG")"
    fi
    BACKFILL="$(grep -E 'Backfill done|Nothing to backfill' "$LOG" | tail -n1 || true)"
    # 2. today's racecard, 3. fast inference (history = raceform + all finished days)
    "$PY" data_collection/fetch_daily_racecards.py --day today 2>&1 | tee "$LOG"
    "$PY" ml/kaggle_v2/fast_inference.py --date "$DAY" 2>&1 | tee -a "$LOG"
    RUNNERS="$(grep -oE 'Upserted [0-9]+ runner rows' "$LOG" | grep -oE '[0-9]+' | head -n1 || true)"
    RACES="$(grep -oE 'predictions \([0-9]+ races\)' "$LOG" | grep -oE '[0-9]+' | head -n1 || true)"
    PICKS="$(grep -E -- '->.*\(p=' "$LOG" | head -n 12 | sed 's/^ *//' || true)"
    notify done "Racecard $DAY fetched: ${RACES:-?} races, ${RUNNERS:-?} runners; inference done" \
      "Backfill: ${BACKFILL:-none}. Top picks (first 12 races):
$PICKS"
    ;;
  score)
    "$PY" ml/kaggle_v2/score_predictions.py --date "$DAY" 2>&1 | tee "$LOG"
    T1="$(grep -oE '^top1_accuracy +=  ?[0-9.]+' "$LOG" | grep -oE '[0-9.]+$' || true)"
    T3="$(grep -oE '^top3_accuracy +=  ?[0-9.]+' "$LOG" | grep -oE '[0-9.]+$' || true)"
    P3="$(grep -oE '^precision_at_3 +=  ?[0-9.]+' "$LOG" | grep -oE '[0-9.]+$' || true)"
    SC="$(grep -oE '^Scored [0-9]+/[0-9]+ races' "$LOG" | head -n1 || true)"
    pct() { awk -v x="${1:-}" 'BEGIN{ if (x=="") print "?"; else printf "%.1f%%", x*100 }'; }
    notify done "Results $DAY: top-1 $(pct "$T1"), top-3 $(pct "$T3"), P@3 $(pct "$P3") (${SC:-scored})" \
      "$(grep -E '^(Scored|top1|top3|precision|mean_position)' "$LOG")"
    ;;
  *)
    echo "usage: run_daily.sh fetch|score" >&2; exit 2 ;;
esac
