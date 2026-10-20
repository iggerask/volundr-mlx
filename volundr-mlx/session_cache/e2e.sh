#!/bin/zsh
# End-to-end session-cache measurement through ../../serve.sh (worker + agens-server), for session_cache/step.sh
# under ../with_gpu_lock.sh (the worker holds the GPU for the whole run):
#   1. the agent conversation with the cache (session_e2e.py run), 2. the same requests with the cache off (replay,
#   budget 0, same worker process), 3. concurrency checks (cache back on), 4. the table.
# usage: session_cache/e2e.sh [outdir]   (from volundr-mlx)
WT=${0:A:h:h}
ROOT=${WT:h}
cd $WT
PY=${PY:-/Users/ingemarrask/personal-dev/llms/.venv/bin/python}
OUT=${1:-session_cache/e2e}
PORT=${PORT:-30123}
mkdir -p $OUT
export VOLUNDR_WORKER_SOCKET=${TMPDIR:-/tmp}/vsc-e2e.sock
export VOLUNDR_WORKER_SOCKET=${VOLUNDR_WORKER_SOCKET//\/\//\/}
[[ ,${PHASES:-run}, == *,run,* ]] && rm -f $OUT/worker.jsonl
echo "### serve.sh $(date '+%T')"
$ROOT/serve.sh --port $PORT --cache-gb 12 --log $OUT/worker.jsonl -- --log-tokens > $OUT/serve.log 2>&1 &
SERVE=$!
cleanup() {
  kill $SERVE 2>/dev/null; sleep 3
  pkill -f "volundr-server --model-dir .* --port $PORT" 2>/dev/null
  pkill -f "worker.py --socket $VOLUNDR_WORKER_SOCKET" 2>/dev/null
  sleep 2
  pkill -9 -f "worker.py --socket $VOLUNDR_WORKER_SOCKET" 2>/dev/null
  true
}
trap cleanup EXIT
for i in {1..300}; do
  curl -sf localhost:$PORT/health >/dev/null && break
  kill -0 $SERVE 2>/dev/null || { echo "serve.sh exited"; tail -30 $OUT/serve.log; exit 1 }
  sleep 2
done
curl -sf localhost:$PORT/health >/dev/null || { echo "server not up"; tail -30 $OUT/serve.log; exit 1 }
grep -E "^worker: (model|warm)" $OUT/serve.log
PHASES=${PHASES:-run,replay,checks}
if [[ ,$PHASES, == *,run,* ]]; then
  echo "### run (cache on) $(date '+%T')"
  $PY session_e2e.py run --port $PORT --out $OUT/cached.json || exit 1
fi
if [[ ,$PHASES, == *,replay,* ]]; then
  echo "### replay (cache off) $(date '+%T')"
  $PY session_e2e.py replay --port $PORT --transcript $OUT/cached.json --out $OUT/nocache.json || exit 1
fi
if [[ ,$PHASES, == *,checks,* ]]; then
  echo "### checks $(date '+%T')"
  $PY session_e2e.py checks --port $PORT --out $OUT/checks.json --budget 12
fi
if [[ -f $OUT/nocache.json ]]; then
  echo "### report $(date '+%T')"
  $PY session_e2e.py report $OUT/cached.json $OUT/nocache.json --worker-log $OUT/worker.jsonl \
    --out $OUT/turns.json | tee $OUT/table.md
fi
curl -s localhost:$PORT/session_cache > $OUT/session_cache_endpoint.json
echo "worker process RSS (KB): $(ps -o rss= -p $(pgrep -f "worker.py --socket $VOLUNDR_WORKER_SOCKET" | head -1) 2>/dev/null)"
grep -c . $OUT/worker.jsonl
