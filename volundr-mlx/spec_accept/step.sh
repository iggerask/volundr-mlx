#!/bin/zsh
# usage: step.sh <log> <cmd...>  (run under ../with_gpu_lock.sh): canary, the command (cwd volundr-mlx), canary, with
# load average and swap around it; all appended to spec_accept/<log>.
PY=/Users/ingemarrask/personal-dev/llms/.venv/bin/python
export PY
WT=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-af7e4848a9651c921/volundr-mlx
cd $WT
log=$WT/spec_accept/$1; shift
{
  echo "### $(date '+%F %T') $*"
  echo "load: $(uptime | sed 's/.*load averages*: //'); swap: $(sysctl -n vm.swapusage)"
  $PY ../agens-mlx-bench/canary.py 2>&1 | grep canary
  "$@" 2>&1 | grep --line-buffered -vE "PyTorch was not found|instantiate a model of type|incorrect regex|^\[transformers\]"
  echo "exit ${pipestatus[1]}"
  $PY ../agens-mlx-bench/canary.py 2>&1 | grep canary
  echo "load: $(uptime | sed 's/.*load averages*: //'); swap: $(sysctl -n vm.swapusage)"
  echo "### done $(date '+%F %T')"
} >> $log 2>&1
