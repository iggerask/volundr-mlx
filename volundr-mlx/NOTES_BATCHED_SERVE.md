Status: in progress (2026-10-20).

# Batched serving: continuous batching in the MLX worker (branch batched-serve)

## Resume

Branch `batched-serve` from `batched-bench` (6d692bb), worktree `.claude/worktrees/agent-a3bd58d97e91bdf21`. Data
links as in the brief (.venv, models, ref, spec/suite.json, spec/runs, parity/*.bin).

Done:
- `batch.py`: slot caches with per-row positions (BatchCache, BatchBCSACache, BatchKVCache, forward,
  capture_row, prebuild / rollback); `volundr.Attention._batched` / `_bcsa_batched`; `nax_hook.min_rows`.
- `scheduler.py` (BatchEngine) wired into `worker.py`; `session.prefill_iter`, `verify.forward_iter`,
  `specdec.SpecStepper` (spec_decode drains it).
- `tests/test_batch.py` 5/5 (tiny model); gates `batched_serve/gate.py exact` (bit-identical at 150-9,000 tokens),
  `engine`, `rows`, `timing` (sync); cargo test 4/4 (new: concurrent streams).

Running / next: timing A/B (sync vs pipelined) + regression chain (`batched_serve/regress.sh`), then the server runs
(`batched_serve/serve_bench.sh`), then this file and the READMEs.
