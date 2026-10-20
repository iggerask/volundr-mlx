#!/bin/zsh
# The existing gates after the session-cache changes (run under ../with_gpu_lock.sh, from volundr-mlx): the unit tests,
# check_combined (verify paths bit-exact, KL at the floor) and, if given, an extra command (e.g. the spec-loop
# equivalence check against main's specdec.py). Logs to session_cache/regress.log.
WT=${0:A:h:h}
cd $WT
S=session_cache/step.sh
PY=${PY:-/Users/ingemarrask/personal-dev/llms/.venv/bin/python}
$S regress.log $PY tests/test_session.py
$S regress.log $PY tests/test_volundr.py
$S regress.log $PY tests/test_decode_fused.py
$S regress.log $PY tests/test_bcsa_verify.py
$S regress.log $PY spec/check_combined.py --ref spec/runs/combined-test_tune_spec_baseline.jsonl
[[ -n $1 ]] && $S regress.log $PY "$@"
true
