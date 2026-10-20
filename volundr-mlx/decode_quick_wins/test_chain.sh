#!/bin/zsh
# Held-out test split, once, final default config (decode_fused on, bf16 lm_head), spec/NOTES.md protocol: the frozen
# spec config, then plain greedy, back to back (one process each, as the protocol's runs).
# Outputs: spec/runs/decode-quick-wins_test_{frozen,base}.jsonl.
cd /Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx/spec
unset VOLUNDR_LMHEAD8 VOLUNDR_FUSED
echo "=== spec $(date +%T)"
$PY specdec.py bench --split test --new 256 --nax --drafter-bits 4 --head-bits 4 --lookup --out runs/decode-quick-wins_test_frozen.jsonl
$PY ../../agens-mlx-bench/canary.py
echo "=== base $(date +%T)"
$PY specdec.py base --split test --new 256 --out runs/decode-quick-wins_test_base.jsonl
