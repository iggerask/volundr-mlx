#!/bin/zsh
# Tune-split interleaved spec A/B (spec/ab_suite.py, frozen config), one process per category group as in
# decode-quick-wins' tune_chain.sh (so each config's Lookup history matches that run's), canary around each process.
# Run under ../with_gpu_lock.sh. Outputs: spec/runs/combined-test_tune_spec_<config>.jsonl (append; --resume).
# usage: tune_chain.sh [group ...]   (default: all three groups)
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-a7157c0891991d4d6/volundr-mlx/combined_test
CF=baseline,exact,exact+q8g64
groups=(coding,humanities,json,math,multilingual,qa,rag reasoning,roleplay,stem,summarization,writing agent,long)
(( $# )) && groups=("$@")
for cats in $groups; do
  $D/step.sh tune_chain.log zsh -c "cd spec && \$PY ab_suite.py spec --split tune --cats $cats --configs $CF --out runs/combined-test_tune_spec --resume"
done
echo "tune chain done $(date '+%T')" >> $D/tune_chain.log
