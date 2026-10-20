#!/bin/zsh
# 8K follow-up: far-block selection per row (kernel) against the ops path at B = 8 / 16 (NAX on), two reps; it also
# repeats the 8K B = 16 timing with a smaller harness snapshot (4 distinct prompts). From volundr-mlx under
# ./with_gpu_lock.sh.
PY=/Users/ingemarrask/personal-dev/llms/.venv/bin/python
cd ${0:A:h:h}
batched/step.sh select.log $PY batched/bench_batched.py --tokens 8192 --batches 1,8,16 --nax 1 --distinct 4 \
  --selects kernel,ops --reps 2 --steps 64 --out spec/runs/batched-bench_decode_8192_select.jsonl
