#!/bin/zsh
# Repeat of tune agent/02 and agent/03 (noisy in the A/B: they ran during the swap-full OOM retries), all three
# configs, same Lookup history (agent/00-01 records seeded into the out files and replayed by --resume). Run under
# ../with_gpu_lock.sh.
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-a7157c0891991d4d6/volundr-mlx/combined_test
$D/step.sh agent_rerun.log zsh -c "cd spec && \$PY ab_suite.py spec --split tune --cats agent --configs baseline,exact,exact+q8g64 --out runs/combined-test_tune_spec_agentrerun --resume"
