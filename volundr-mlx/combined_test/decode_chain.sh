#!/bin/zsh
# Plain greedy decode A/B (bench_decode.py, configs interleaved per rep on one loaded model), short and 19K context.
# Run under ../with_gpu_lock.sh.
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-a7157c0891991d4d6/volundr-mlx/combined_test
CF=baseline,exact,exact+q8g64
$D/step.sh decode.log zsh -c "\$PY bench_decode.py --model models/volundr-int4-mlx --configs $CF --reps 3 --tokens 128"
$D/step.sh decode.log zsh -c "\$PY bench_decode.py --model models/volundr-int4-mlx --prompt-file bench/long_prompt.txt --no-think --shared-cache --configs $CF --reps 3 --tokens 48"
echo "decode chain done $(date '+%T')" >> $D/decode.log
