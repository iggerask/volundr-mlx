#!/bin/zsh
# usage: step.sh <log> <cmd...>  (run under ../with_gpu_lock.sh): canary, the command (cwd volundr-mlx), canary, with
# the load average around it; all appended to combined_test/<log>.
PY=/Users/ingemarrask/personal-dev/llms/.venv/bin/python
export PY
WT=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-a7157c0891991d4d6/volundr-mlx
cd $WT
log=$WT/combined_test/$1; shift
{
  echo "### $(date '+%F %T') $*"
  echo "load: $(uptime | sed 's/.*load averages*: //')"
  $PY ../agens-mlx-bench/canary.py 2>&1 | grep canary
  "$@" 2>&1 | grep --line-buffered -vE "PyTorch was not found|instantiate a model of type|incorrect regex|^\[transformers\]"
  echo "exit ${pipestatus[1]}"
  $PY ../agens-mlx-bench/canary.py 2>&1 | grep canary
  echo "load: $(uptime | sed 's/.*load averages*: //')"
  echo "### done $(date '+%F %T')"
} >> $log 2>&1
