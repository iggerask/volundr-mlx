#!/bin/zsh
# Step 0 gates, second attempt of the interleaved check (the first stopped on a bug in the state comparison). The old
# unsynchronised prefill peaks at ~56 GB and pushes other apps into swap, so long/03 runs it once only.
# Run under ../with_gpu_lock.sh.
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-af7e4848a9651c921/volundr-mlx
S=$D/spec_accept/step.sh
$S step0.log zsh -c "VOLUNDR_LMHEAD8=64 \$PY spec_accept/step0_check.py --names long/03-long3 --reps 1 --out spec/runs/spec-accept_step0_check.jsonl"
$S step0.log zsh -c "VOLUNDR_LMHEAD8=64 \$PY spec_accept/step0_check.py --names agent/01-pytest-dev__iniconfig,coding/00-12623519-think,rag/00-330006df-think,json/00-boolq2-think --reps 2 --out spec/runs/spec-accept_step0_check.jsonl"
echo "step0b done $(date '+%T')" >> $D/spec_accept/step0.log
