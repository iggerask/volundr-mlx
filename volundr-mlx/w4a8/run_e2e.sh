#!/bin/zsh
# End-to-end prefill A/B (bf16 path vs VOLUNDR_W4A8=1), interleaved, canary before / after. Run under ../with_gpu_lock.sh.
P=${PY:-${0:A:h}/../../.venv/bin/python}
cd "${0:A:h}/.."
$P ../agens-mlx-bench/canary.py
for i in 1 2; do
  echo "== bf16 (stock) run $i"; $P bench_prefill.py --model models/volundr-int4-mlx --tokens 20480 --decode 16 2>&1 | grep -v "transformers\]"
  echo "== W4A8 run $i"; VOLUNDR_W4A8=1 $P bench_prefill.py --model models/volundr-int4-mlx --tokens 20480 --decode 16 2>&1 | grep -v "transformers\]"
done
$P ../agens-mlx-bench/canary.py
