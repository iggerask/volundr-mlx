#!/bin/zsh
# bcsa-verify-8q: held-out test split, once, frozen config (8q BCSA verify), plus greedy and the previous config.
# usage: test_chain.sh 8q|prefill|base  (run from volundr-mlx/spec under the GPU lock, via gpurun.sh)
PYB=/Users/ingemarrask/personal-dev/llms/.venv/bin/python
CAN=../../agens-mlx-bench/canary.py
FROZEN=(--nax --drafter-bits 4 --head-bits 4 --lookup)
R=runs/bcsa-verify-8q_
canary() { $PYB $CAN 2>&1 | grep canary }
filt() { grep --line-buffered -vE "PyTorch was not found|instantiate a model of type|incorrect regex|^\[transformers\]" }
echo "### test $1 $(date)"; canary
if [[ $1 == base ]]; then
  $PYB specdec.py base --split test --new 256 --out ${R}test_base.jsonl 2>&1 | filt | tail -2
else
  VOLUNDR_BCSA_VERIFY=$1 $PYB specdec.py bench --split test --new 256 $FROZEN --out ${R}test_$1.jsonl 2>&1 | filt | tail -2
fi
canary
echo "### done $1 $(date)"
