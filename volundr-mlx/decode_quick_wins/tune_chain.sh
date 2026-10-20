#!/bin/zsh
# Tune-split interleaved A/Bs (spec/ab_suite.py), spec then greedy per category group, canary around each process.
# Outputs: spec/runs/decode-quick-wins_tune_{spec,greedy}_<config>.jsonl (append).
cd /Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx/spec
CF=main,fused,fused+q8g64,fused+q8g128
for cats in coding,humanities,json,math,multilingual,qa,rag reasoning,roleplay,stem,summarization,writing agent,long; do
  for mode in spec greedy; do
    echo "=== $mode $cats $(date +%T)"
    $PY ../../agens-mlx-bench/canary.py
    $PY ab_suite.py $mode --split tune --cats $cats --configs $CF --out runs/decode-quick-wins_tune_$mode --resume
    $PY ../../agens-mlx-bench/canary.py
  done
done
