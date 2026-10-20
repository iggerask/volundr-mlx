#!/bin/zsh
# Quick schedule A/B (lever 3 candidates, one prompt per category), the batched-verify correctness check and its cost.
# Run under ../with_gpu_lock.sh.
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-af7e4848a9651c921/volundr-mlx
S=$D/spec_accept/step.sh
$S sched_quick.log zsh -c "cd spec && \$PY ab_suite.py spec --split tune --per-cat 1 --configs exact+q8g64,exact+q8g64@early,exact+q8g64@s2 --out runs/spec-accept_sched_quick --resume"
$S b2.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY check_verify_b2.py --nax --cats coding,json,agent,long"
$S b2.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY time_b2.py --names coding/00-12623519-think,long/01-long1"
echo "next1 done $(date '+%T')" >> $D/spec_accept/b2.log
