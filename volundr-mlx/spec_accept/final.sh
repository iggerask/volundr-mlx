#!/bin/zsh
# After the tune A/B: the short held-out sample (test split, coding + long + agent, 18 prompts), baseline then the final
# config back to back, once; the step breakdown with the final config; the B = 1 verify gate (check_verify, bf16 head,
# against combined_test/check_verify.log). Run under ../with_gpu_lock.sh.
# usage: final.sh <loop flags of the final config, e.g. early@cv@long16>
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-af7e4848a9651c921/volundr-mlx
S=$D/spec_accept/step.sh
L=$1
A=(bench --split test --cats coding,long,agent --new 256 --nax --drafter-bits 4 --head-bits 4 --lookup --resume)
$S heldout.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY specdec.py $A --out runs/spec-accept_test_baseline.jsonl"
$S heldout.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY specdec.py $A --loop $L --out runs/spec-accept_test_final.jsonl"
echo "heldout done $(date '+%T')" >> $D/spec_accept/heldout.log
$S prof.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY prof_step.py --compiled --names coding/00-12623519-think,agent/01-pytest-dev__iniconfig,long/01-long1 --new 160 --out runs/spec-accept_prof_cv.jsonl"
$S prof.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY prof_step.py --compiled --long 16 --names coding/00-12623519-think,agent/01-pytest-dev__iniconfig,long/01-long1 --new 160 --out runs/spec-accept_prof_cv_long16.jsonl"
$S gates.log zsh -c "cd spec && \$PY check_verify.py --nax"
echo "final done $(date '+%T')" >> $D/spec_accept/gates.log
