#!/bin/zsh
# Attribution of decode-quick-wins' +1.5 GB spec-prefill peak (peak_probe.py, long/03, fresh process per config):
# unfused (decode_fused off; views, wide norms, cached norm weights on) and fused+copyidx (views off). Compare with
# combined-baseline (= main) and combined-exact (= fused) in peak.log. Run under ../with_gpu_lock.sh.
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-a7157c0891991d4d6/volundr-mlx/combined_test
C=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-a7157c0891991d4d6/volundr-mlx
probe() { $D/step.sh peak.log zsh -c "$1 \$PY $D/peak_probe.py $2" }
probe "" "--label combined-unfused --tree $C --config unfused"
probe "" "--label combined-fused+copyidx --tree $C --config fused+copyidx"
echo "peak chain3 done $(date '+%T')" >> $D/peak.log
