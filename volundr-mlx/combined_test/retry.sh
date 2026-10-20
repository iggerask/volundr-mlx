#!/bin/zsh
# usage: retry.sh <max_attempts> tune|test [chain args...]   (run under ../with_gpu_lock.sh)
# Re-runs a --resume chain (tune_chain.sh / test_chain.sh) until complete.py reports all records present or the
# attempts run out: the Metal OOMs under system memory pressure (NOTES_COMBINED.md) stop a process mid-run, and
# --resume with the Lookup replay continues it as if uninterrupted.
D=/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-a7157c0891991d4d6/volundr-mlx/combined_test
max=$1; stage=$2; shift 2
for i in $(seq 1 $max); do
  echo "### retry.sh $stage attempt $i/$max $(date '+%T'); swap: $(sysctl -n vm.swapusage)" >> $D/retry.log
  $D/${stage}_chain.sh "$@"
  if /Users/ingemarrask/personal-dev/llms/.venv/bin/python $D/complete.py $stage >> $D/retry.log; then
    echo "### retry.sh $stage complete after attempt $i $(date '+%T')" >> $D/retry.log
    exit 0
  fi
done
echo "### retry.sh $stage gave up after $max attempts $(date '+%T')" >> $D/retry.log
exit 1
