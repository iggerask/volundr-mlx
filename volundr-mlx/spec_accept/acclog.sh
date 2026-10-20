#!/bin/zsh
# Step 3 measurement: per-step acceptance of alternative chains on the tune split (spec/accept_log.py), baseline config
# (exact + VOLUNDR_LMHEAD8=64), one process, --check against spec_generate. Run under ../with_gpu_lock.sh.
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-af7e4848a9651c921/volundr-mlx
S=$D/spec_accept/step.sh
$S acclog.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY accept_log.py --split tune --check --resume --out runs/spec-accept_acclog_tune.jsonl"
echo "acclog done $(date '+%T')" >> $D/spec_accept/acclog.log
