#!/bin/zsh
# Memory probe (mem_probe.py) with canaries; run under ../with_gpu_lock.sh. Args: [prompt name] [configs].
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-a7157c0891991d4d6/volundr-mlx/combined_test
$D/step.sh mem_probe.log zsh -c "cd spec && \$PY ../combined_test/mem_probe.py $*"
