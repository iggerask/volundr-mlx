#!/bin/zsh
# Server-mode measurements through ../../serve.sh (batching worker + agens-server), for batched_serve/step.sh under
# ../with_gpu_lock.sh (the worker holds the GPU for the whole run):
#   1. the server-mode run (19 prompts x 256 tokens at concurrency 1, 2, 4, 8, 16), 2. the mixed run (agent session +
#   short streams), then, with the server stopped, 3. completions against concurrency 1 and 4. the tie check.
# usage: batched_serve/serve_bench.sh [outdir] [extra worker.py flags]   (from volundr-mlx)
#   env PHASES (default server,mixed,compare,ties), LEVELS (default 1,2,4,8,16), PORT
WT=${0:A:h:h}
ROOT=${WT:h}
cd $WT
PY=${PY:-/Users/ingemarrask/personal-dev/llms/.venv/bin/python}
OUT=${1:-batched_serve/run}
shift 2>/dev/null
PORT=${PORT:-30124}
PHASES=${PHASES:-server,mixed,compare,ties}
mkdir -p $OUT
export VOLUNDR_WORKER_SOCKET=${TMPDIR:-/tmp}/vbs-bench.sock
export VOLUNDR_WORKER_SOCKET=${VOLUNDR_WORKER_SOCKET//\/\//\/}
if [[ ,$PHASES, == *,server,* || ,$PHASES, == *,mixed,* ]]; then
  echo "### serve.sh $(date '+%T')"
  $ROOT/serve.sh --port $PORT --cache-gb 12 --log $OUT/worker.jsonl -- --log-tokens "$@" > $OUT/serve.log 2>&1 &
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
  if [[ ,$PHASES, == *,server,* ]]; then
    echo "### server-mode run $(date '+%T')"
    $PY batched_serve/serve_bench.py server --port $PORT --worker-log $OUT/worker.jsonl --out $OUT/server.jsonl \
      --levels ${LEVELS:-1,2,4,8,16} --canary || exit 1
  fi
  if [[ ,$PHASES, == *,mixed,* ]]; then
    echo "### mixed run $(date '+%T')"
    $PY batched_serve/serve_bench.py mixed --port $PORT --worker-log $OUT/worker.jsonl --out $OUT/mixed.jsonl || exit 1
  fi
  curl -s localhost:$PORT/session_cache > $OUT/session_cache_endpoint.json
  echo "worker RSS (KB): $(ps -o rss= -p $(pgrep -f "worker.py --socket $VOLUNDR_WORKER_SOCKET" | head -1) 2>/dev/null)"
  cleanup
  trap - EXIT
fi
if [[ ,$PHASES, == *,compare,* ]]; then
  echo "### compare $(date '+%T')"
  $PY batched_serve/serve_bench.py compare --worker-log $OUT/worker.jsonl --run $OUT/server.jsonl --out $OUT/compare.jsonl
fi
if [[ ,$PHASES, == *,ties,* ]]; then
  echo "### ties $(date '+%T')"
  $PY batched_serve/serve_bench.py ties --compare $OUT/compare.jsonl --out $OUT/ties.jsonl
fi
true
