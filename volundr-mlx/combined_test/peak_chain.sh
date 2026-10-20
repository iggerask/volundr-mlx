#!/bin/zsh
# Single-config, fresh-process peak memory on long/03-long3 (peak_probe.py): combined (this branch) vs main and the two
# branches (read-only `git archive` exports in the scratchpad), spec command-buffer settings vs MLX's defaults, spec vs
# plain greedy. Run under ../with_gpu_lock.sh; one canary pair per probe (step.sh).
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-a7157c0891991d4d6/volundr-mlx/combined_test
C=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-a7157c0891991d4d6/volundr-mlx
T=/private/tmp/claude-501/-Users-ingemarrask-personal-dev-llms/b08d8102-657c-4e37-b76d-c8780a9ddb09/scratchpad/trees
probe() { $D/step.sh peak.log zsh -c "$1 \$PY $D/peak_probe.py $2" }
probe "VOLUNDR_LMHEAD8=64" "--label combined-exact+q8g64 --tree $C"
probe "" "--label combined-baseline --tree $C --config baseline"
probe "" "--label combined-exact --tree $C"
probe "" "--label main --tree $T/main/volundr-mlx"
probe "" "--label bcsa-verify-8q --tree $T/bcsa/volundr-mlx"
probe "" "--label decode-quick-wins --tree $T/dqw/volundr-mlx"
probe "VOLUNDR_LMHEAD8=64" "--label combined-exact+q8g64-defaultbufs --tree $C --bufs default"
probe "" "--label main-defaultbufs --tree $T/main/volundr-mlx --bufs default"
probe "VOLUNDR_LMHEAD8=64" "--label combined-greedy-exact+q8g64 --tree $C --mode greedy --bufs default"
echo "peak chain done $(date '+%T')" >> $D/peak.log
