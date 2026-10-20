#!/bin/zsh
# bcsa-verify-8q: free-running check, then tune-split A/B (8q vs prefill BCSA verify) and a greedy baseline.
# Run from volundr-mlx/spec under the GPU lock (gpurun.sh does both); canary around every run.
PYB=/Users/ingemarrask/personal-dev/llms/.venv/bin/python
CAN=../../agens-mlx-bench/canary.py
FROZEN=(--nax --drafter-bits 4 --head-bits 4 --lookup)
R=runs/bcsa-verify-8q_
canary() { $PYB $CAN 2>&1 | grep canary }
filt() { grep --line-buffered -vE "PyTorch was not found|instantiate a model of type|incorrect regex|^\[transformers\]" }
stage=${1:-all}
if [[ $stage == a || $stage == all ]]; then
if [[ ! -s ${R}check_frozen.jsonl ]]; then  # done once (14 prompts); the run file appends
echo "### check (free-running, 8q)"; canary
VOLUNDR_BCSA_VERIFY=8q $PYB specdec.py check --per-cat 1 --new 128 $FROZEN --out ${R}check_frozen.jsonl 2>&1 | filt
canary
fi
echo "### tune 8q"; canary
VOLUNDR_BCSA_VERIFY=8q $PYB specdec.py bench --split tune --new 256 $FROZEN --out ${R}tune_8q.jsonl 2>&1 | filt | tail -3
canary
echo "### tune prefill"; canary
VOLUNDR_BCSA_VERIFY=prefill $PYB specdec.py bench --split tune --new 256 $FROZEN --out ${R}tune_prefill.jsonl 2>&1 | filt | tail -3
canary
fi
if [[ $stage == b || $stage == all ]]; then
echo "### tune greedy"; canary
$PYB specdec.py base --split tune --new 256 --out ${R}tune_base.jsonl 2>&1 | filt | tail -3
canary
fi
if [[ $stage == c || $stage == all ]]; then
for rep in 2 3; do
  for i in prefill 8q; do
    echo "### long+agent $i rep $rep"; canary
    VOLUNDR_BCSA_VERIFY=$i $PYB specdec.py bench --split tune --cats long,agent --new 256 $FROZEN --out ${R}tune_la_${i}_r$rep.jsonl 2>&1 | filt | tail -2
    canary
  done
done
fi
echo "### done $stage $(date)"
