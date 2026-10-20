#!/bin/zsh
# usage: gpu_run.sh <log> <cmd...>   (run under with_gpu_lock.sh): canary, command (cwd volundr-mlx), canary
PY=/Users/ingemarrask/personal-dev/llms/.venv/bin/python
export PY
WT=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx
cd $WT
log=$1; shift
{
  echo "### $(date '+%F %T') $*"
  $PY ../agens-mlx-bench/canary.py
  "$@" 2>&1 | grep -vE "PyTorch was not found|instantiate a model of type|incorrect regex"
  echo "exit ${pipestatus[1]}"
  $PY ../agens-mlx-bench/canary.py
  echo "### done $(date '+%F %T')"
} >> $log 2>&1
