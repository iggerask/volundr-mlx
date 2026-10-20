#!/bin/zsh
# The full-model gates (batched_serve/gate.py) in one lock session, from volundr-mlx: rows, timing (short and 8K).
# usage: ./with_gpu_lock.sh batched_serve/gates.sh [rows,timing,timing8k,exact,engine]
WT=${0:A:h:h}
cd $WT
S=batched_serve/step.sh
PY=${PY:-/Users/ingemarrask/personal-dev/llms/.venv/bin/python}
G=${1:-rows,timing}
[[ ,$G, == *,exact,* ]] && $S gate_exact.log $PY batched_serve/gate.py exact --out batched_serve/gate_exact.jsonl
[[ ,$G, == *,rows,* ]] && $S gate_rows.log $PY batched_serve/gate.py rows --lengths 120,300,800,1500,3000,4500,6000,9000 \
  --steps 96 --out batched_serve/gate_rows.jsonl
[[ ,$G, == *,timing,* ]] && $S gate_timing.log $PY batched_serve/gate.py timing --tokens 200 --out batched_serve/gate_timing.jsonl
[[ ,$G, == *,timing8k,* ]] && $S gate_timing.log $PY batched_serve/gate.py timing --tokens 8192 --batches 1,2,4,8 \
  --out batched_serve/gate_timing.jsonl
[[ ,$G, == *,engine,* ]] && $S gate_engine.log $PY batched_serve/gate.py engine --out batched_serve/gate_engine.jsonl
true
