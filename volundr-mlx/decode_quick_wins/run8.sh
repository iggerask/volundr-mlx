#!/bin/zsh
# exactness (tests + real-model parity dump vs parity/mlx.bin), then plain-decode A/Bs at short, 14K and 24K context.
# The 24K prompt: the first 88000 characters of ref/agens-engine/kernels/m1/m1_model.cu wrapped like bench/long_prompt.txt
# (bench.sh), kept outside the repo.
cd /Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx
L24=/private/tmp/claude-501/-Users-ingemarrask-personal-dev-llms/b08d8102-657c-4e37-b76d-c8780a9ddb09/scratchpad/long24k_prompt.txt
$PY tests/test_decode_fused.py
$PY dump_logits.py --model models/volundr-int4-mlx --prompts parity/prompts.json --out parity/dqw_fused2.bin --tokens-out parity/dqw_fused2.json
cmp parity/dqw_fused2.bin parity/mlx.bin && echo "FUSED2 DUMP == mlx.bin (bit-identical)"
$PY bench_decode.py --model models/volundr-int4-mlx --configs unfused+copyidx,fused+copyidx,fused,fused+q8g64,fused+q8g128 --reps 3 --tokens 128
$PY bench_decode.py --model models/volundr-int4-mlx --prompt-file bench/long_prompt.txt --no-think --shared-cache --configs unfused+copyidx,fused+copyidx,fused,fused+q8g64 --reps 3 --tokens 48
$PY bench_decode.py --model models/volundr-int4-mlx --prompt-file $L24 --no-think --shared-cache --configs unfused+copyidx,fused+copyidx,fused,fused+q8g64 --reps 3 --tokens 48
