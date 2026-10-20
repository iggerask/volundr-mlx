#!/bin/zsh
# usage: batched/step.sh <log> <cmd...>  (run under ../with_gpu_lock.sh): canary, the command (cwd volundr-mlx),
# canary, with load average and swap around it; all appended to batched/<log>.
PY=${PY:-/Users/ingemarrask/personal-dev/llms/.venv/bin/python}
export PY
WT=${0:A:h:h}
cd $WT
log=$WT/batched/$1; shift
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
