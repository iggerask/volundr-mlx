# Volundr MLX

An MLX port of [Agens Volundr 32B](https://huggingface.co/Blockway/Agens-Volundr-32B-v1-INT4) (Blockway) for Apple
silicon. The model is a hybrid: 54 Kimi Delta Attention layers, 17 BCSA sparse-attention layers and one dense
attention layer, with Engram n-gram memory and 4-stream mHC residuals. The upstream engine is CUDA-only
([agens-engine](https://github.com/BlockWayz/agens-engine)). Everything here was built and measured on an M5 Pro
(64 GB, 20-core GPU) with MLX 0.32.4.

It serves an OpenAI-compatible API through the upstream agens-server front end, with an MLX worker behind it:

- DFlash2 speculative decoding for greedy requests;
- a session / prefix cache, so a follow-up turn only prefills its new tokens;
- continuous batching of up to 16 concurrent requests.

Unofficial; not affiliated with Blockway.

| Folder | What it is |
|---|---|
| [`volundr-mlx/`](volundr-mlx/README.md) | Converter, model, prefill kernels, opt-in W4A8 prefill ([`w4a8/`](volundr-mlx/w4a8/NOTES.md)), DFlash2 speculative decoding ([`spec/`](volundr-mlx/spec/NOTES.md)), the MLX worker with its session cache and continuous batching, parity tooling ([`PARITY.md`](volundr-mlx/PARITY.md)) |
| [`server/`](server/) | `volundr-server`: the upstream agens-server crate (OpenAI API, chat template, tool-call and reasoning parsers) with a backend that forwards to the MLX worker over a Unix socket |
| [`agens-mlx-bench/`](agens-mlx-bench/) | Synthetic Volundr-shaped benchmarks, and the small-batch 4-bit matmul on the M5 GPU neural accelerators used for speculative verify ([`smallm/`](agens-mlx-bench/smallm/NOTES.md)) |

## Setup

You need:

- An Apple silicon Mac with enough unified memory. The weights take 24 GB resident, and one request peaks at
  25–35 GB. Batched serving with a long-context request reached 53 GB. Only a 64 GB machine has been tested.
- Python 3.12.
- Rust 1.87 or newer (`cargo`), for the server. The first `./serve.sh` builds it.
- About 23 GB of disk for the converted model, plus the Hugging Face downloads.

```bash
git clone https://github.com/iggerask/volundr-mlx && cd volundr-mlx
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/hf download Blockway/Agens-Volundr-32B-v1-INT4 --exclude "assets/*"
.venv/bin/hf download Blockway/Agens-Volundr-32B-v1-DFlash2
.venv/bin/python volundr-mlx/convert.py --hf-path Blockway/Agens-Volundr-32B-v1-INT4 \
  --mlx-path volundr-mlx/models/volundr-int4-mlx
```

- The conversion takes about 40 s. It repacks the INT4 weights without re-quantizing them, so every dequantized
  weight is bit-identical to the release.
- The DFlash2 drafter is read from the default Hugging Face cache (`~/.cache/huggingface/hub`).
- The scripts expect the virtualenv at `.venv`. To use another interpreter, set `PY=...`.
- Model weights aren't included. They're covered by their Hugging Face repos' licenses.

## Use

### Serve the OpenAI API

```bash
./serve.sh
```

- This starts the MLX worker and `volundr-server` on `127.0.0.1:30000`.
- The server starts listening once the worker has loaded the model and warmed up, which takes about a minute.
- Ctrl-C stops both.

Send greedy requests (`"temperature": 0`) to get speculative decoding. The model's `generation_config.json` defaults
to temperature 1.0 with top-k 20, and sampled requests decode without speculation.

```bash
curl -s localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write a Python function that merges two sorted lists."}],
  "temperature": 0, "max_tokens": 512}'
```

| Request field or header | Effect |
|---|---|
| `"stream": true` | Server-sent events, as in the OpenAI API |
| `"chat_template_kwargs": {"enable_thinking": false}` | Turns thinking off (it is on by default). The thinking is returned separately in `reasoning_content`. |
| `X-Session-Id: <id>` header, or `"session_id": "<id>"` | Names the conversation, so its next turn resumes from the cached end of the last reply. Requests without one still reuse the longest cached prefix. |
| `"tools": [...]` | Tool calls, parsed by the upstream front end |

| Endpoint | What it is |
|---|---|
| `POST /v1/chat/completions`, `POST /v1/completions` | Generation |
| `GET /v1/models`, `GET /health` | Model list and health check |
| `GET /session_cache` | The worker's cache: snapshots, bytes and recent requests |

Options for `serve.sh`: `--port 30000`, `--host 127.0.0.1`, `--cache-gb 12` (the session-cache budget; 0
turns it off) and `--log FILE` (one JSON record per request). Worker flags go after `--`, for example
`./serve.sh -- --max-batch 8 --short-warmup`.

| Worker flag | Default | What it does |
|---|---|---|
| `--max-batch` | 16 | Requests active at once, prefilling or decoding |
| `--slice-ms` | 200 | Prefill time between two decode steps while other requests are decoding |
| `--max-active-gb` | 46 | Admits no new request while MLX holds more than this, unless nothing else is active |
| `--bspec-max` | 4 | Batched speculation while 2 to N greedy requests decode |
| `--no-spec`, `--no-bspec` | | Decode greedy requests without speculation, alone or in a batch |
| `--context-length` | 262144 | Longest prompt plus reply |
| `--short-warmup` | | Skips the 5K-token warm-up, so the server is ready sooner. The first long prompt then pays the kernel compile. |

Opt-in modes, set in the environment:

- `VOLUNDR_LMHEAD8=64` runs the output layer in 8 bits, group size 64. Decoding is 4–7% faster. Some outputs
  differ from the bf16 head: 47 of 56 greedy test replies came out identical.
- `VOLUNDR_W4A8=1` runs prefill with int8 activations: 25–32% faster at 20K tokens, with slightly different
  numerics ([`w4a8/NOTES.md`](volundr-mlx/w4a8/NOTES.md)).

For example: `VOLUNDR_LMHEAD8=64 ./serve.sh`.

### Generate from the command line

```bash
.venv/bin/python volundr-mlx/generate.py --model volundr-mlx/models/volundr-int4-mlx \
  --prompt "Hello" --max-tokens 256 [--no-think]
```

This is plain greedy decoding without the drafter or the cache. It prints the text, tok/s and peak memory. For
speculative decoding, use the server.

### Tests

```bash
cd volundr-mlx
for t in test_volundr test_decode_fused test_bcsa_verify test_session test_batch; do ../.venv/bin/python tests/$t.py; done
cd ../server && cargo test
```

The Python tests use tiny models against a NumPy fp64 port of the release's torch code. They don't need the
converted model.

## Performance

M5 Pro (64 GB, 307 GB/s), INT4 weights, MLX 0.32.4. Warm runs, with an idle GPU checked before each run (a
26–28 TFLOP/s matmul canary).

### One request

| | |
|---|---|
| Greedy decode, speculative (the server's default for `temperature 0`) | 27.6 tok/s on 56 tune prompts, 2.6 tokens per verify step; the first release's code does 24.5 on the same prompts |
| Greedy decode, plain | 13.8 tok/s (14.7 with `VOLUNDR_LMHEAD8=64`) |
| Decode step at 19K context | 73.4 ms (69.0 with the 8-bit head; 78.5 before the decode fusion) |
| Prefill, exact | 374 tok/s on a 19K-token prompt |
| Prefill, `VOLUNDR_W4A8=1` | 490–509 tok/s at 20K |
| Peak memory | ~25 GB at short context, ~29 GB at 19K; ~28 GB speculative on a 12.5K-token prompt |

The speculative figures were measured with the opt-in 8-bit head, which gives about 4% of the gain. Speculative speed
depends on how predictable the text is:

| Prompt category (tune split) | tok/s | tokens per step |
|---|---|---|
| Agent sessions (4.5–7.3K context) | 43.2 | 4.2 |
| JSON | 43.1 | 4.0 |
| Math | 42.7 | 4.0 |
| Coding | 32.7 | 3.0 |
| Reasoning | 32.7 | 3.1 |
| Summarization | 27.2 | 2.6 |
| Writing, roleplay, STEM, RAG, humanities, multilingual | 24–26 | 2.3–2.4 |
| QA | 22.6 | 2.1 |
| Long prompts (8–12.5K) | 21.2 | 2.1 |

### Serving concurrent requests

Through `./serve.sh` with default settings, run the way upstream measures server mode: 19 short prompts, 256 new
tokens each, greedy, thinking off. TTFT is the time from sending a request to its first streamed token.

| Concurrent requests | Aggregate tok/s | Per request tok/s | TTFT median / p95 | Peak memory |
|---|---|---|---|---|
| 1 | 24.0 | 25.0 | 0.24 / 0.25 s | 34.5 GB |
| 2 | 38.0 | 20.1 | 0.25 / 0.55 s | 34.5 GB |
| 4 | 56.1 | 15.3 | 0.25 / 0.94 s | 35.4 GB |
| 8 | 64.2 | 10.0 | 0.53 / 2.58 s | 36.0 GB |
| 16 | 86.2 | 8.1 | 2.51 / 6.42 s | 39.0 GB |

- One greedy request uses the speculative loop. Two to four use batched speculation, and five or more use plain
  batched decoding.
- The single-request figure is below 27.6 because the prompts differ and the server runs the default bf16 head.
- At 16 requests the rate is 129.8 tok/s while all 16 are generating. The average is lower because the last few
  requests finish alone.
- Replies that differ from the single-request run all differ at near-ties between the top two tokens. Batched
  matmuls sum in a different order, so exact ties can resolve differently.
- With a long prompt arriving while short requests decode, prefill runs in 200 ms slices between decode steps. The
  short requests keep streaming at a reduced rate instead of stalling; without slicing they stalled for 14 s.
  There is still one stall of about 4 s when the long request joins the batch.

### Follow-up turns (session cache)

| Turn | Time to first token, cached | Without the cache |
|---|---|---|
| 21.7K-token context, 3.5K new tokens | 9.1 s | 52.7 s |
| 18K-token context, 1.2K new tokens | 3.1 s | 43.6 s |

The cache holds snapshots of the model state at the end of each prompt and reply, and every 4,096 tokens of a long
prefill. It is kept under a 12 GB budget, with least recently used snapshots evicted first.

### Compared with the CUDA engine

Upstream's own figures (the [agens-engine README](https://github.com/BlockWayz/agens-engine)): the 24 GB profile with
the same 4-bit weights and drafter, on a 48 GB NVIDIA GPU with about 3× the M5 Pro's memory bandwidth.

| | Agens Engine (CUDA) | This port (M5 Pro) |
|---|---|---|
| Coding, one request | 112.8 tok/s | 32.7 |
| Reasoning, one request | 101.3 | 32.7 |
| Agent session, one request | 120.6 (8K context) | 43.2 (4.5–7.3K) |
| JSON output, one request | 199.4 | 43.1 |
| Server mode, 1 / 2 / 4 concurrent | 87.7 / 129.3 / 155.1 | 24.0 / 38.0 / 56.1 |

The prompts aren't the same, so treat the rows as rough comparisons. Both sides accept about the same number of
draft tokens per step. The gap is in the cost of a verify step: Apple's GPU has much less compute per unit of
bandwidth. This port gains more from batching (×2.3 from one to four requests, against ×1.8 upstream), because a
single stream here is far from compute-bound.

### Compared with llama.cpp on the same Mac

Stock llama.cpp, mlx-lm and vLLM don't support the Volundr architecture. Blockway's
[llama.cpp fork](https://github.com/BlockWayz/llama.cpp/tree/volundr) (branch `volundr`) is the only other engine
that runs it on Apple silicon. Below is its
[Q4_K_M GGUF](https://huggingface.co/Blockway/Agens-Volundr-32B-v1-GGUF) on the same M5 Pro, with Metal and all
layers on the GPU, at two fork commits:

- **bb1c8bc**, the updated commit and the branch tip as of 2026-10-20. Blockway added Metal kernels for the Volundr
  ops in 3ea6ab6 and bb1c8bc.
- **67e4001**, the commit tested first. These are the earlier measurements, kept for comparison.

The same settings, prompts and scripts were used for both commits.

| | Fork at bb1c8bc (updated) | Fork at 67e4001 (first test) | This port, INT4 |
|---|---|---|---|
| Decode, short context | 9.8 tok/s | 8.6 tok/s | 13.8 plain, 27.6 speculative |
| Decode at 1K context | 9.3 tok/s | 8.1 tok/s | — |
| Decode at 8K context | 7.4 tok/s | 6.0 tok/s | — |
| Decode at 19K context | 7.7 tok/s | 5.3 tok/s | 13.6 plain |
| Prefill, 512-token prompt | 294 tok/s | 226 tok/s | — |
| Prefill, 19K-token prompt | 137 tok/s | 117 tok/s | 374 tok/s |
| What runs on the CPU | Only the token-embedding lookup | Every mHC and Engram op: 298 GPU/CPU splits per forward pass | — |
| Speculative decoding | No | No | Yes |
| GPU memory | 31.4 GB, with 32K of context preallocated | 31.3 GB, the same way | ~29 GB at 19K |

- **What changed between the commits.** At 67e4001, the mHC and Engram ops had no Metal kernels, so they ran on the
  CPU in every layer. With the kernels in bb1c8bc:
  - Decode is 14–43% faster.
  - Short-prompt prefill is 26–30% faster, and the 19K prompt prefills 17% faster.
  - The output is unchanged: greedy replies are byte-identical between the two commits.
- **Where the fork's decode time goes.** At bb1c8bc, decode takes about 102 ms per token; at 67e4001 it took 117 ms.
  Reading the weights once takes about 73 ms at this bandwidth, and this port's plain step is 72 ms. Where the fork's
  remaining ~30 ms goes hasn't been profiled.
- **Output quality.** At both commits, the output is coherent and matches llama.cpp's own CPU backend.
- **Why the replies differ from this port's.** Greedy replies from the two engines diverge within the first two dozen
  tokens. That's expected from a different 4-bit quantization: Q4_K_M keeps some layers at 8 bits, and this port uses
  Blockway's INT4 with a bf16 head.
- **Details.** The method, both runs and the raw results are in [`NOTES_LLAMACPP.md`](volundr-mlx/NOTES_LLAMACPP.md).

## Status

Experimental. Known limits:

- Speculative decoding is greedy only; sampled requests decode plainly.
- Batched speculation doesn't help past four concurrent requests.
- Each batch slot is sized to the longest request in the batch, so one long-context request raises memory for all.
- The session cache lives in memory only; it doesn't spill to disk.
- The CUDA-side parity run ([`PARITY.md`](volundr-mlx/PARITY.md) section 3) hasn't been done. The tests check the
  model against a NumPy fp64 port of the release's torch code.

Benchmarks, the parity tools and the prompt-suite builder read upstream reference files from `volundr-mlx/ref/`,
which isn't tracked. To recreate it, run `volundr-mlx/fetch_ref.sh`.

## License

The code is MIT-licensed ([`LICENSE`](LICENSE)). [`NOTICE`](NOTICE) lists the third-party work it builds on:
agens-engine (MIT), the Volundr release code (Apache 2.0) and MLX (MIT). Model weights aren't included; they're
covered by the licenses in their Hugging Face repos.
