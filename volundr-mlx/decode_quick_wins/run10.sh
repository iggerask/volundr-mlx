#!/bin/zsh
# exactness (tests + real-model parity dump vs parity/mlx.bin), then plain-decode A/Bs (ab_modes configs) at short, 14K
# and 24K context. The 24K prompt: the first 88000 characters of ref/agens-engine/kernels/m1/m1_model.cu wrapped as
# bench.sh wraps bench/long_prompt.txt; kept outside the repo (scratchpad).
cd /Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx
L24=/private/tmp/claude-501/-Users-ingemarrask-personal-dev-llms/b08d8102-657c-4e37-b76d-c8780a9ddb09/scratchpad/long24k_prompt.txt
$PY tests/test_decode_fused.py
$PY tests/test_volundr.py 2>&1 | tail -3
$PY dump_logits.py --model models/volundr-int4-mlx --prompts parity/prompts.json --out parity/dqw_fused3.bin --tokens-out parity/dqw_fused3.json
cmp parity/dqw_fused3.bin parity/mlx.bin && echo "FUSED3 DUMP == mlx.bin (bit-identical)"
$PY bench_decode.py --model models/volundr-int4-mlx --configs main,unfused,fused+copyidx,fused,fused+q8g64,fused+q8g128 --reps 3 --tokens 128
$PY ../agens-mlx-bench/canary.py
$PY bench_decode.py --model models/volundr-int4-mlx --prompt-file bench/long_prompt.txt --no-think --shared-cache --configs main,fused+copyidx,fused,fused+q8g64 --reps 3 --tokens 48
$PY ../agens-mlx-bench/canary.py
$PY bench_decode.py --model models/volundr-int4-mlx --prompt-file $L24 --no-think --shared-cache --configs main,fused+copyidx,fused,fused+q8g64 --reps 3 --tokens 48
