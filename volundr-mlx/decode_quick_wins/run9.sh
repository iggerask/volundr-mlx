#!/bin/zsh
# exactness of the wide (staged) row kernels, then the chain benchmark of NT / wide 512 / wide 1024
cd /Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx
$PY tests/test_decode_fused.py
$PY decode_quick_wins/chain3.py
