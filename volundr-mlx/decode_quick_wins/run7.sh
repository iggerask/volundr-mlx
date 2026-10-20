#!/bin/zsh
# exactness tests for the constant-trip-count row loops, then the old-vs-unrolled norm / mHC chain microbenchmark
cd /Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx
$PY tests/test_decode_fused.py
$PY tests/test_volundr.py
$PY decode_quick_wins/chain2.py
