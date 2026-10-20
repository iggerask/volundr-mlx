#!/bin/zsh
# Plain batched decode, short and 8K context (run from volundr-mlx under ./with_gpu_lock.sh). CTXS="200" for one.
PY=/Users/ingemarrask/personal-dev/llms/.venv/bin/python
cd ${0:A:h:h}
for ctx in ${=CTXS:-200 8192}; do
  batched/step.sh decode.log $PY batched/bench_batched.py --tokens $ctx --batches 1,2,4,8,16 --distinct 8 --steps 128 \
    --out spec/runs/batched-bench_decode_$ctx.jsonl
done
