#!/bin/zsh
# Second lock session (from volundr-mlx, under ./with_gpu_lock.sh): the e2e concurrency checks again on a fresh
# worker (the first run left the cache budget at 0 from the replay), the tie check of the e2e replies, then the
# regression chain. Each step logs canaries, load and swap.
WT=${0:A:h:h}
cd $WT
PY=/Users/ingemarrask/personal-dev/llms/.venv/bin/python
PHASES=checks session_cache/step.sh e2e_checks.log session_cache/e2e.sh session_cache/e2e
session_cache/step.sh tie_check.log $PY session_cache/tie_check.py session_cache/e2e/cached.json \
  session_cache/e2e/nocache.json --out session_cache/e2e/tie_check.json
session_cache/regress.sh
