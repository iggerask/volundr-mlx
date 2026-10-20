Status: in progress (2026-10-20).

# Batched serving: continuous batching in the MLX worker (branch batched-serve)

## Resume

Branch `batched-serve` from `batched-bench` (6d692bb), worktree `.claude/worktrees/agent-a3bd58d97e91bdf21`. Data
links as in the brief (.venv, models, ref, spec/suite.json, spec/runs, parity/*.bin).

Done:
- `batch.py`: slot caches with per-row positions (BatchCache, BatchBCSACache, BatchKVCache, forward,
  capture_row); `volundr.Attention._batched` / `_bcsa_batched`; `nax_hook.min_rows`.
- `tests/test_batch.py` 4/4 (tiny model, GPU): B = 1 batch == decode_step bit for bit; rows joining / leaving at
  different lengths == their B = 1 runs bit for bit; buffer growth; slots.

Next: prefill / spec steppers, the scheduler in worker.py, full-model gates, server runs.
