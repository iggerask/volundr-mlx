"""Shared helpers for the W4A8 prefill spike: GPU-lock-aware timing (reuses agens-mlx-bench/smallm/common.py)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "agens-mlx-bench" / "smallm"))
import common as _smallm  # noqa: E402
from common import BF, LOCK, bench, compare, spin_up, wait_gpu_idle  # noqa: E402,F401


def _ancestors():
    import subprocess
    out = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True).stdout
    parent = dict(tuple(int(v) for v in line.split()) for line in out.splitlines() if line.strip())
    pid, seen = os.getpid(), []
    while pid in parent and pid > 1:
        pid = parent[pid]
        seen.append(pid)
    return seen


def _held_by_ancestor():
    try:
        return int(open(LOCK).read().strip()) in _ancestors()
    except (OSError, ValueError):
        return False


# Under with_gpu_lock.sh an ancestor (the wrapper) holds the lock; the in-process wait would wait on it forever.
if _held_by_ancestor():
    _smallm.wait_gpu_idle = lambda poll=2.0: None

from w4a8 import MPP_HEADER  # noqa: E402,F401

# (name, N, K) of the prefill matmuls; gate/up are separate 17408 x 5120 linears in volundr.py, 34816 is the pair.
SHAPES = [("gate_up", 34816, 5120), ("gate", 17408, 5120), ("down", 5120, 17408), ("q_proj", 12288, 5120),
          ("kda_qkv", 10240, 5120), ("z", 6144, 5120), ("out", 5120, 6144)]
