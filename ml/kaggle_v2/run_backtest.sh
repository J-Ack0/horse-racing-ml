#!/usr/bin/env bash
# Run the whole-day backtest, evaluate whatever finished, then push a phone note.
#   run_backtest.sh [extra backtest_gap.py args]
# Resumable: days that already have a predictions CSV are skipped, so rerunning after
# freeing memory / enabling swap only redoes the days that were OOM-killed.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/venv/bin/python"
OUT="$HERE/predictions/backtest"
NOTIFY="$HOME/.local/bin/notify-phone.sh"

nice -n 10 "$PY" -u "$HERE/backtest_gap.py" --strided "$@"
RC=$?
N=$(ls "$OUT"/predictions_*.csv 2>/dev/null | wc -l)
FAILED=$(cat "$OUT/failures.txt" 2>/dev/null | wc -l)
if [ "$N" -eq 0 ]; then
  "$NOTIFY" error "Backtest finished with no predictions (rc=$RC, $FAILED days failed)" "Most likely OOM: a full day needs ~5 GB. See $OUT and the driver log." || true
  exit 1
fi
"$PY" "$HERE/evaluate_backtest.py" > "$OUT/evaluate.log" 2>&1
ERC=$?
if [ $ERC -ne 0 ]; then
  "$NOTIFY" error "Backtest: $N days predicted but evaluation failed" "$(tail -n 12 "$OUT/evaluate.log")" || true
  exit 1
fi
T1=$(grep -m1 'top-1 (pick won)' "$OUT/report.md" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
T3=$(grep -m1 'top-3 (pick placed)' "$OUT/report.md" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
P3=$(grep -m1 'precision@3' "$OUT/report.md" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
AP=$(grep -m1 'average precision' "$OUT/report.md" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
BODY="$(sed -n '/^## Headline/,/^## Runner-level/p' "$OUT/report.md" | head -n 16)
Failed days (OOM, need more memory): $(tr '\n' ' ' < "$OUT/failures.txt" 2>/dev/null)
Full report: $OUT/report.md"
"$NOTIFY" done "Backtest done: $N days, top-1 $T1, top-3 $T3, P@3 $P3, AP $AP ($FAILED days failed)" "$BODY" || true
