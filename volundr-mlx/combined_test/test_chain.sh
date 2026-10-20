#!/bin/zsh
# Short held-out sample: test split, coding + long + agent (18 prompts), frozen spec config, baseline then
# exact+q8g64 back to back. baseline needs --config (main's decode path incl. copying BCSA indexing, prefill verify);
# exact+q8g64 is the env-selected default with VOLUNDR_LMHEAD8=64. Run under ../with_gpu_lock.sh.
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-a7157c0891991d4d6/volundr-mlx/combined_test
A=(bench --split test --cats coding,long,agent --new 256 --nax --drafter-bits 4 --head-bits 4 --lookup --resume)
$D/step.sh test_chain.log zsh -c "cd spec && \$PY specdec.py $A --config baseline --out runs/combined-test_test_baseline.jsonl"
$D/step.sh test_chain.log zsh -c "cd spec && VOLUNDR_LMHEAD8=64 \$PY specdec.py $A --out runs/combined-test_test_exact+q8g64.jsonl"
echo "test chain done $(date '+%T')" >> $D/test_chain.log
