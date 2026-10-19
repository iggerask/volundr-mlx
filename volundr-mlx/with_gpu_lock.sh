#!/bin/zsh
# Run a command holding the shared GPU lock (same protocol as bench.sh / agens-mlx-bench/run_clean.sh).
LOCK="${GPU_LOCK:-$HOME/Library/Application Support/Claude/side-session-notes/local_4b1c7a56-3804-4800-b519-29795572f128/gpu.lock}"
others_running() {  # another venv python process (matched on the executable, not on command lines)
  ps -axo pid=,comm=,args= | awk -v me=$$ '{c=$2; sub(".*/","",c)} c ~ /^python/ && $0 ~ /personal-dev\/llms/ && $0 !~ /multiprocessing/ && $1 != me {f=1} END {exit !f}'
}
lock_busy() { [[ -f "$LOCK" ]] || return 1; local h=$(<"$LOCK"); [[ -n "$h" && "$h" != "$$" ]] && kill -0 "$h" 2>/dev/null }
release() { [[ -f "$LOCK" && "$(<"$LOCK")" == "$$" ]] && rm -f "$LOCK" }
trap release EXIT INT TERM
while lock_busy || others_running; do [[ -z "$a" ]] && { echo "[waiting for the GPU to go idle]"; a=1 }; sleep 2; done
mkdir -p "${LOCK:h}" && print -n $$ > "$LOCK"
"$@"
