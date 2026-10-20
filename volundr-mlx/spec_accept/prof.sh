#!/bin/zsh
# Step 1 profile (spec/prof_step.py at short and ~10K context, time_parts verify widths) and the step-3 acceptance log
# (spec/accept_log.py, tune split). Baseline config: exact + VOLUNDR_LMHEAD8=64. Run under ../with_gpu_lock.sh.
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-af7e4848a9651c921/volundr-mlx
S=$D/spec_accept/step.sh
$S prof.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY prof_step.py --names coding/00-12623519-think,agent/01-pytest-dev__iniconfig,long/01-long1 --new 160 --out runs/spec-accept_prof_base.jsonl"
$S prof.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY time_parts.py --ctx short --drafter-bits 4 --head-bits 4 --nax --Ts 8,16 --reps 6"
$S prof.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY time_parts.py --ctx long --drafter-bits 4 --head-bits 4 --nax --Ts 8,16 --reps 6"
echo "prof done $(date '+%T')" >> $D/spec_accept/prof.log
