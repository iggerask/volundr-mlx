"""The worker's warm-up alone (no socket): load, warm up, print the engine's step counts, exit.
usage (from volundr-mlx, under ../with_gpu_lock.sh): python batched_serve/warmup_check.py [worker.py flags]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import worker  # noqa: E402

sys.argv = ["worker.py", "--short-warmup"] + sys.argv[1:]
import argparse  # noqa: E402

orig = argparse.ArgumentParser.parse_args
args = None


def grab(self, *a, **k):
    global args
    args = orig(self, *a, **k)
    raise SystemExit(0)


argparse.ArgumentParser.parse_args = grab
try:
    worker.main()
except SystemExit:
    pass
argparse.ArgumentParser.parse_args = orig
w = worker.Worker(args)
print(f"warm-up {w.warm_up():.1f} s", flush=True)
print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in w.engine.stats.items()}), flush=True)
