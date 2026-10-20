#!/bin/zsh
# Tune-split interleaved spec A/B (spec/ab_suite.py, frozen config), one process per category group as in
# combined_test/tune_chain.sh, canary around each process. Run under ../with_gpu_lock.sh.
# usage: tune_chain.sh <out prefix under spec/runs> <configs> [group ...]   (default: all three groups)
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-af7e4848a9651c921/volundr-mlx/spec_accept
OUT=$1; CF=$2; shift 2
groups=(coding,humanities,json,math,multilingual,qa,rag reasoning,roleplay,stem,summarization,writing agent,long)
(( $# )) && groups=("$@")
for cats in $groups; do
  $D/step.sh tune_chain.log zsh -c "cd spec && \$PY ab_suite.py spec --split tune --cats $cats --configs $CF --out runs/$OUT --resume"
done
echo "tune chain $OUT done $(date '+%T')" >> $D/tune_chain.log
