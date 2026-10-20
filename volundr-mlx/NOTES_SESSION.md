# Session / prefix cache and serving through agens-server (branch session-cache, 2026-10-20)

Status: in progress (stage 4 running).

## Resume

- Branch `session-cache` from main (66c00f2), in a clone of the llms repo at
  `/Users/ingemarrask/personal-dev/llms/volundr-mlx/ref/agens-engine/.claude/worktrees/agent-ab9a6f81b8d270d5c/volundr-mlx-session-cache`
  (the harness put this agent's worktree in the agens-engine repo; the coordinator fetches `session-cache` from the
  clone). Data links as in the brief (.venv, models, ref, spec/suite.json, spec/runs, parity/*.bin).
- Done: stage 1 (session.py, tests/test_session.py, session_gate.py gates 1-4), stage 2 (worker.py, sampler.py),
  stage 3 (server/, serve.sh; `cargo build --release` and `cargo test` pass).
- Running / next: stage 4 (`session_cache/e2e.sh` under the GPU lock), then the regression chain
  (`session_cache/regress.sh`), then this file's tables.
