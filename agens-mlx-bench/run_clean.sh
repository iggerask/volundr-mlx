#!/bin/zsh
# Runs the Volundr benchmarks with a GPU canary before and after each script.
# Other sessions benchmark on this GPU too: wait while gpu.lock is held by a live PID or another
# venv python job runs, then hold the lock for the whole batch (same protocol as smallm/common.py).
cd "${0:A:h}"
PY=${PY:-${0:A:h}/../.venv/bin/python}
LOCK="${GPU_LOCK:-$HOME/Library/Application Support/Claude/side-session-notes/local_4b1c7a56-3804-4800-b519-29795572f128/gpu.lock}"

others_running() {
  ps -axo pid=,command= | grep '\.venv/bin/python' | grep -v multiprocessing | grep -v grep | grep -q .
}
lock_busy() {
  [[ -f "$LOCK" ]] || return 1
  local holder=$(<"$LOCK")
  [[ -n "$holder" && "$holder" != "$$" ]] && kill -0 "$holder" 2>/dev/null
}
release() { [[ -f "$LOCK" && "$(<"$LOCK")" == "$$" ]] && rm -f "$LOCK" }
trap release EXIT INT TERM

while lock_busy || others_running; do
  [[ -z "$announced" ]] && { echo "[waiting for the GPU to go idle]"; announced=1 }
  sleep 2
done
mkdir -p "${LOCK:h}" && print -n $$ > "$LOCK"

run() {
  echo "### $*"
  $PY canary.py 2>&1 | grep canary
  $PY "$@" 2>&1 | grep -vE 'Warning|^\s*$' | grep -vE '^peak'
  $PY canary.py 2>&1 | grep canary
}
run volundr_synth.py 4 128 1:short:linears 1:short:tuned 5:short:linears 5:short:tuned 8:short:linears 8:short:tuned
run volundr_synth.py 4 128 2048:short:linears 2048:short:tuned
run prefill_parts.py
run prefill_long.py
