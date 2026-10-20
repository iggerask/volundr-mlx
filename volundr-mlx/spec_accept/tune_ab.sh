#!/bin/zsh
# The tune-split A/B of the levers (all 56 prompts, interleaved per prompt, groups as in combined_test), re-run with
# --resume until complete (at most 3 attempts). Run under ../with_gpu_lock.sh.
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-af7e4848a9651c921/volundr-mlx/spec_accept
OUT=spec-accept_tune
CF=exact+q8g64,exact+q8g64@early@cv,exact+q8g64@early@cv@long16,exact+q8g64@early@cv@long16c
for i in 1 2 3; do
  echo "### tune_ab attempt $i $(date '+%T'); swap: $(sysctl -n vm.swapusage)" >> $D/tune_chain.log
  $D/tune_chain.sh $OUT $CF
  /Users/ingemarrask/personal-dev/llms/.venv/bin/python $D/complete.py $OUT $CF tune >> $D/tune_chain.log && break
done
echo "tune_ab done $(date '+%T')" >> $D/tune_chain.log
