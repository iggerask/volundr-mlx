#!/bin/zsh
# Blockway's llama.cpp fork (branch volundr) on Metal, Q4_K_M GGUF: llama-bench prompt processing and decode at
# depth 0, ~1K and ~8K, all layers offloaded. Run it under ../with_gpu_lock.sh; canary before and after.
#   usage: ../with_gpu_lock.sh ./bench_llama.sh [llama-bench flags; they replace the default matrix below]
cd "${0:A:h}"
LL=${LL:-$HOME/personal-dev/llama.cpp-volundr/build/bin}
M=${M:-$HOME/personal-dev/models/volundr-gguf/Agens-Volundr-32B-v1-Q4_K_M.gguf}
PY=${PY:-$HOME/personal-dev/llms/.venv/bin/python}
canary() { $PY ../../agens-mlx-bench/canary.py 2>&1 | grep canary }
(( $# )) || set -- -r 3 -p 512,2048 -n 64 -d 0,1024,8192
canary
/usr/bin/time -l $LL/llama-bench -m $M -ngl 99 -o md "$@" 2>&1 \
  | grep -vE "^(llama_|load|print_info|ggml_metal_|\.)" | grep -vE "^\s*$"
canary
