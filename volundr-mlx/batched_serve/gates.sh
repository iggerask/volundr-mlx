#!/bin/zsh
# The full-model gates (batched_serve/gate.py) in one lock session, from volundr-mlx: rows, timing (short and 8K).
# usage: ./with_gpu_lock.sh batched_serve/gates.sh [rows,timing,timing8k,exact,engine,bspec,serve,prof,bspecn,tests,profeng]
WT=${0:A:h:h}
cd $WT
S=batched_serve/step.sh
PY=${PY:-/Users/ingemarrask/personal-dev/llms/.venv/bin/python}
G=${1:-rows,timing}
[[ ,$G, == *,tests,* ]] && $S tests.log $PY tests/test_batch.py
[[ ,$G, == *,tests,* ]] && $S tests.log $PY tests/test_session.py
[[ ,$G, == *,exact,* ]] && $S gate_exact.log $PY batched_serve/gate.py exact --out batched_serve/gate_exact.jsonl
[[ ,$G, == *,rows,* ]] && $S gate_rows.log $PY batched_serve/gate.py rows --lengths 120,300,800,1500,3000,4500,6000,9000 \
  --steps 96 --out batched_serve/gate_rows.jsonl
[[ ,$G, == *,timing,* ]] && $S gate_timing.log $PY batched_serve/gate.py timing --tokens 200 --out batched_serve/gate_timing.jsonl
[[ ,$G, == *,timing8k,* ]] && $S gate_timing.log $PY batched_serve/gate.py timing --tokens 8192 --batches 1,2,4,8 \
  --out batched_serve/gate_timing.jsonl
[[ ,$G, == *,engine,* ]] && $S gate_engine.log $PY batched_serve/gate.py engine --out batched_serve/gate_engine.jsonl
[[ ,$G, == *,bspec,* ]] && $S gate_bspec.log $PY batched_serve/gate.py bspec --lengths 300,5200 --steps 24 --new 128 \
  --out batched_serve/gate_bspec.jsonl
[[ ,$G, == *,serve,* ]] && $S gate_serve.log $PY batched_serve/gate.py serve --batches ${SERVE_LEVELS:-16,2,4} \
  --bspec-ab 1 --out batched_serve/gate_serve.jsonl
[[ ,$G, == *,prof,* ]] && $S gate_prof.log $PY batched_serve/gate.py prof --batches 8,16 --steps 32 --new 128 \
  --out batched_serve/gate_prof.jsonl
[[ ,$G, == *,bspecn,* ]] && $S gate_bspecn.log $PY batched_serve/gate.py bspecn --batches 2,3,4,6,8 --new 128 \
  --out batched_serve/gate_bspecn.jsonl
[[ ,$G, == *,profeng,* ]] && $S gate_prof.log $PY batched_serve/gate.py prof --batches 16 --steps 0 --engines 1 \
  --new 128 --out batched_serve/gate_prof.jsonl
true
