#!/bin/zsh
# Third lock session (from volundr-mlx, under ./with_gpu_lock.sh): the full e2e again with the server's disconnect
# handling and the worker's token log, the tie check on that log, and the chunk-invariance probe with the KDA
# kernels split out.
WT=${0:A:h:h}
cd $WT
PY=/Users/ingemarrask/personal-dev/llms/.venv/bin/python
session_cache/step.sh e2e.log session_cache/e2e.sh session_cache/e2e
session_cache/step.sh tie_check.log $PY session_cache/tie_check.py session_cache/e2e/worker.jsonl \
  --out session_cache/e2e/tie_check.json
session_cache/step.sh gate5.log $PY session_cache/chunk_invariance.py
