#!/bin/bash
# Full pipeline. Run from this directory.
set -x
PY=/home/alarm/venvs/horse-racing-ml/bin/python
cd "$(dirname "$0")"
$PY exp_final.py            > final_ukire.log 2>&1
$PY exp_rank_retry.py       > rank_retry.log  2>&1
$PY exp_tune.py --ire --obj softmax --n 10 > tune_ire_softmax.log 2>&1
$PY exp_tune.py --ire --obj binary  --n 10 > tune_ire_binary.log  2>&1
$PY exp_final.py --ire      > final_ire.log 2>&1
echo ALLDONE
