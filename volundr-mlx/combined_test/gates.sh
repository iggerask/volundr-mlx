#!/bin/zsh
# Correctness gates for the merge, each step with a canary around it (step.sh). Run under ../with_gpu_lock.sh.
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-a7157c0891991d4d6/volundr-mlx/combined_test
S=$D/step.sh
PY=/Users/ingemarrask/personal-dev/llms/.venv/bin/python
export PY
$S gates.log $PY tests/test_volundr.py
$S gates.log $PY tests/test_decode_fused.py
$S gates.log $PY tests/test_bcsa_verify.py
$S gates.log zsh -c '$PY dump_logits.py --model models/volundr-int4-mlx --prompts parity/prompts.json --out parity/combined-test_default.bin --tokens-out parity/combined-test_default.json && cmp parity/combined-test_default.bin parity/mlx.bin && echo "DEFAULT DUMP == parity/mlx.bin (bit-identical)"'
$S check_verify.log zsh -c 'cd spec && $PY check_verify.py --nax'
$S check_combined.log zsh -c 'cd spec && $PY check_combined.py --nax'
echo "gates chain done $(date '+%T')" >> $D/gates.log
