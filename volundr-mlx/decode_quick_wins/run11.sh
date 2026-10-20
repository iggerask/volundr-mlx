#!/bin/zsh
# plain-decode A/B at ~23.6K context (the first 63500 characters of m1_model.cu, wrapped as bench.sh does; outside the repo)
cd /Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx
L23=/private/tmp/claude-501/-Users-ingemarrask-personal-dev-llms/b08d8102-657c-4e37-b76d-c8780a9ddb09/scratchpad/long23k_prompt.txt
$PY bench_decode.py --model models/volundr-int4-mlx --prompt-file $L23 --no-think --shared-cache --configs main,fused+copyidx,fused,fused+q8g64 --reps 4 --tokens 48
