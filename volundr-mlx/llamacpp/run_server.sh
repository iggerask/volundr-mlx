#!/bin/zsh
# Start one server, send prompts.json through chat.py (greedy, thinking off), stop the server. Run it under
# ../with_gpu_lock.sh so only one GPU job runs at a time.
#   usage: ../with_gpu_lock.sh ./run_server.sh llama|mlx OUT.json [chat.py flags, e.g. --only fib,sky]
#   env (llama): LLAMA_EXTRA="..." extra llama-server flags; GGML_SCHED_DEBUG etc. pass through
cd "${0:A:h}"
which=$1 out=$2; shift 2
LL=${LL:-$HOME/personal-dev/llama.cpp-volundr/build/bin}
M=${M:-$HOME/personal-dev/models/volundr-gguf/Agens-Volundr-32B-v1-Q4_K_M.gguf}
PY=${PY:-$HOME/personal-dev/llms/.venv/bin/python}
mkdir -p logs
$PY ../../agens-mlx-bench/canary.py 2>&1 | grep canary
if [[ $which == llama ]]; then
  port=8081 url=http://127.0.0.1:8081/health log=logs/${out:r}.server.log
  /usr/bin/time -l $LL/llama-server -m $M -ngl 99 --jinja -c 32768 -np 1 --port $port ${=LLAMA_EXTRA} > $log 2>&1 &
  SPID=$!
  stop() { pkill -TERM -x llama-server; wait }
else
  port=30000 url=http://127.0.0.1:30000/session_cache log=logs/${out:r}.server.log
  ../../serve.sh --port $port > $log 2>&1 &
  SPID=$!
  stop() { kill $SPID; wait }
fi
trap stop EXIT
until curl -sf $url > /dev/null; do
  sleep 1; kill -0 $SPID 2>/dev/null || { echo "server exited"; tail -20 $log; exit 1 }
done
echo "server up after ${SECONDS}s"
curl -s http://127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages": [{"role": "user", "content": "Hi"}], "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": false}}' > /dev/null
/usr/bin/python3 -I ./chat.py --port $port --out $out "$@"
$PY ../../agens-mlx-bench/canary.py 2>&1 | grep canary
