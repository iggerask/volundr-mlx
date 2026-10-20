#!/bin/zsh
# usage: gpurun.sh <cmd...> : runs from volundr-mlx/spec under the GPU lock with canary before/after
W=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ab8462c67ceba885a/volundr-mlx
export PY=/Users/ingemarrask/personal-dev/llms/.venv/bin/python
cd $W/spec
$W/with_gpu_lock.sh zsh -c '
  $PY ../../agens-mlx-bench/canary.py 2>&1 | grep canary
  "$@" 2>&1 | grep --line-buffered -vE "PyTorch was not found|instantiate a model of type|incorrect regex|^\[transformers\]"
  $PY ../../agens-mlx-bench/canary.py 2>&1 | grep canary
' _ "$@"
