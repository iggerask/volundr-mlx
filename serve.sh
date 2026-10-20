#!/bin/zsh
# Serve Volundr MLX through agens-server: start the MLX worker (volundr-mlx/worker.py, the model + session cache) on a
# Unix socket, then the Rust front end (server/, the upstream agens-server crate + MlxBackend) on HTTP. Ctrl-C stops
# both. The server waits for the worker to load (~30 s) and warm up (~20 s).
#
# usage: ./serve.sh [--port 30000] [--host 127.0.0.1] [--cache-gb 12] [--log FILE] [-- extra worker.py flags]
#   env: PY (python, default .venv/bin/python), MODEL (model dir), VOLUNDR_WORKER_SOCKET, VOLUNDR_SESSION_CACHE_GB,
#        VOLUNDR_LMHEAD8 (opt-in 8-bit lm_head), VOLUNDR_W4A8 (opt-in W4A8 prefill)
# then:  curl -s localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -H 'X-Session-Id: s1' \
#          -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
#        curl -s localhost:30000/session_cache    (snapshots, bytes, recent requests)
set -e
ROOT=${0:A:h}
PY=${PY:-$ROOT/.venv/bin/python}
MODEL=${MODEL:-$ROOT/volundr-mlx/models/volundr-int4-mlx}
SOCK=${VOLUNDR_WORKER_SOCKET:-${TMPDIR:-/tmp}/volundr-mlx-worker.sock}
SOCK=${SOCK//\/\//\/}
HOST=127.0.0.1
PORT=30000
wargs=()
while (( $# )); do
  case $1 in
    --port) PORT=$2; shift 2 ;;
    --host) HOST=$2; shift 2 ;;
    --cache-gb) wargs+=(--cache-gb $2); shift 2 ;;
    --log) wargs+=(--log $2); shift 2 ;;
    --) shift; wargs+=("$@"); break ;;
    *) echo "unknown flag $1 (worker flags go after --)"; exit 2 ;;
  esac
done
BIN=$ROOT/server/target/release/volundr-server
if [[ ! -x $BIN ]]; then
  (cd $ROOT/server && cargo build --release)
fi
$PY $ROOT/volundr-mlx/worker.py --socket $SOCK $wargs &
WPID=$!
trap 'kill $WPID 2>/dev/null; wait $WPID 2>/dev/null' EXIT INT TERM
$BIN --model-dir $MODEL --socket $SOCK --host $HOST --port $PORT
