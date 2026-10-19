#!/bin/zsh
# Volundr MLX decode/prefill benchmark under the shared GPU protocol (agens-mlx-bench/run_clean.sh): wait while
# gpu.lock is held by a live PID or another venv python job runs, hold the lock, canary before and after each run.
cd "${0:A:h}"
PY=${PY:-${0:A:h}/../.venv/bin/python}
LOCK="${GPU_LOCK:-$HOME/Library/Application Support/Claude/side-session-notes/local_4b1c7a56-3804-4800-b519-29795572f128/gpu.lock}"
MODEL=${MODEL:-models/volundr-int4-mlx}
others_running() {  # another venv python process (matched on the executable, not on command lines)
  ps -axo pid=,comm=,args= | awk -v me=$$ '{c=$2; sub(".*/","",c)} c ~ /^python/ && $0 ~ /personal-dev\/llms/ && $0 !~ /multiprocessing/ && $1 != me {f=1} END {exit !f}'
}
lock_busy() { [[ -f "$LOCK" ]] || return 1; local h=$(<"$LOCK"); [[ -n "$h" && "$h" != "$$" ]] && kill -0 "$h" 2>/dev/null }
release() { [[ -f "$LOCK" && "$(<"$LOCK")" == "$$" ]] && rm -f "$LOCK" }
trap release EXIT INT TERM
while lock_busy || others_running; do [[ -z "$a" ]] && { echo "[waiting for the GPU to go idle]"; a=1 }; sleep 2; done
mkdir -p "${LOCK:h}" && print -n $$ > "$LOCK"
canary() { $PY ../agens-mlx-bench/canary.py 2>&1 | grep canary }
mkdir -p bench
run() {
  local name=$1; shift
  echo "### $name"; canary
  $PY generate.py --model $MODEL --warmup 16 --json bench/$name.json "$@" 2>&1 | grep -vE "PyTorch was not found|instantiate a model of type|incorrect regex" | tail -c 1500
  canary
}
$PY - <<'PY'
from pathlib import Path
src = Path("ref/agens-engine/kernels/m1/m1_model.cu").read_text()
Path("bench/long_prompt.txt").write_text("Here is a CUDA source file:\n\n```cuda\n" + src[:52000] +
    "\n```\n\nSummarize in five bullet points what this file does.")
PY
run short-think --prompt "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring." --max-tokens 256
run long-14k --prompt-file bench/long_prompt.txt --no-think --max-tokens 192
