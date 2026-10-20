#!/bin/zsh
# The existing gates plus tests/test_batch.py after the batched-serve changes (run under ../with_gpu_lock.sh, from
# volundr-mlx): unit tests, check_combined (verify paths bit-exact, KL at the floor) and, if given, an extra command.
# Logs to batched_serve/regress.log.
WT=${0:A:h:h}
cd $WT
S=batched_serve/step.sh
PY=${PY:-/Users/ingemarrask/personal-dev/llms/.venv/bin/python}
$S regress.log $PY tests/test_batch.py
$S regress.log $PY tests/test_session.py
$S regress.log $PY tests/test_volundr.py
$S regress.log $PY tests/test_decode_fused.py
$S regress.log $PY tests/test_bcsa_verify.py
$S regress.log $PY spec/check_combined.py --ref spec/runs/combined-test_tune_spec_baseline.jsonl
[[ -n $1 ]] && $S regress.log $PY "$@"
true
