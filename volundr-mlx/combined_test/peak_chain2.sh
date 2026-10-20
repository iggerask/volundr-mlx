#!/bin/zsh
# Second half of peak_chain.sh: the archived trees (which failed first for want of agens-mlx-bench/smallm next to them),
# and the ab_suite set-up (three configs' heads in one process) for comparison. Run under ../with_gpu_lock.sh.
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-a7157c0891991d4d6/volundr-mlx/combined_test
C=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-a7157c0891991d4d6/volundr-mlx
T=/private/tmp/claude-501/-Users-ingemarrask-personal-dev-llms/b08d8102-657c-4e37-b76d-c8780a9ddb09/scratchpad/trees
probe() { $D/step.sh peak.log zsh -c "$1 \$PY $D/peak_probe.py $2" }
probe "" "--label main --tree $T/main/volundr-mlx"
probe "" "--label bcsa-verify-8q --tree $T/bcsa/volundr-mlx"
probe "" "--label decode-quick-wins --tree $T/dqw/volundr-mlx"
probe "" "--label main-defaultbufs --tree $T/main/volundr-mlx --bufs default"
probe "" "--label combined-ab_suite-setup-exact+q8g64 --tree $C --modes baseline,exact,exact+q8g64 --config exact+q8g64"
echo "peak chain2 done $(date '+%T')" >> $D/peak.log
