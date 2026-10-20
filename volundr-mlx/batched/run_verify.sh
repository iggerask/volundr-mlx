#!/bin/zsh
# Batched spec step parts (verify forward of B independent sequences x 8 rows, drafter at B), short and 8K context, in
# the spec config's heads (8-bit target head g64, 4-bit drafter + head copy). From volundr-mlx under ./with_gpu_lock.sh.
PY=/Users/ingemarrask/personal-dev/llms/.venv/bin/python
cd ${0:A:h:h}
export VOLUNDR_LMHEAD8=64
for ctx in ${=CTXS:-200 8192}; do
  batched/step.sh verify.log $PY batched/bench_verify.py --tokens $ctx --batches 1,2,4 --reps 10 \
    --out spec/runs/batched-bench_verify.jsonl
done
