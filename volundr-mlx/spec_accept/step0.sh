#!/bin/zsh
# Step 0 (prefill memory fix) gates. Run under ../with_gpu_lock.sh.
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-af7e4848a9651c921/volundr-mlx
S=$D/spec_accept/step.sh
# peak on long/03-long3, fresh process, spec buffers, 8-bit head: layer-synced prefill (new default), then the old one
$S step0.log zsh -c "VOLUNDR_LMHEAD8=64 \$PY combined_test/peak_probe.py --label spec-accept-layersync --tree $D"
$S step0.log zsh -c "VOLUNDR_LMHEAD8=64 VOLUNDR_PREFILL_SYNC=0 \$PY combined_test/peak_probe.py --label spec-accept-nosync --tree $D"
# interleaved prefill speed / bit-identity / spec step time
$S step0.log zsh -c "VOLUNDR_LMHEAD8=64 \$PY spec_accept/step0_check.py --names long/03-long3,agent/01-pytest-dev__iniconfig,coding/00-12623519-think,rag/00-330006df-think --reps 2 --out spec/runs/spec-accept_step0_check.jsonl"
echo "step0 done $(date '+%T')" >> $D/spec_accept/step0.log
