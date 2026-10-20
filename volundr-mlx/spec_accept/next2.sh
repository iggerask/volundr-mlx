#!/bin/zsh
# Compiled KDA verify layers: bit-identity + build time, cProfile of the verify build; long-copy (T = 16) verify
# correctness (teacher-forced, and spec vs greedy); quick schedule A/B with the compiled build. Run under
# ../with_gpu_lock.sh.
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-af7e4848a9651c921/volundr-mlx
S=$D/spec_accept/step.sh
$S compile.log zsh -c "VOLUNDR_LMHEAD8=64 \$PY spec_accept/check_compile.py"
$S compile.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY prof_vbuild.py --reps 20"
$S long16.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY check_verify.py --nax --T 16"
$S long16.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY specdec.py check --per-cat 1 --new 128 --nax --drafter-bits 4 --head-bits 4 --lookup --out runs/spec-accept_check_base.jsonl"
$S long16.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY specdec.py check --per-cat 1 --new 128 --nax --drafter-bits 4 --head-bits 4 --lookup --loop long16 --out runs/spec-accept_check_long16.jsonl"
$S sched_quick.log zsh -c "cd spec && \$PY ab_suite.py spec --split tune --per-cat 1 --configs exact+q8g64,exact+q8g64@cv,exact+q8g64@early@cv --out runs/spec-accept_cv_quick --resume"
echo "next2 done $(date '+%T')" >> $D/spec_accept/compile.log
