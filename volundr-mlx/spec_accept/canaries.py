"""Canary before / after every step.sh step in the spec_accept logs: one line per step (start time, log, command head,
canaries, exit code).
usage: python canaries.py [log ...]   (default: every *.log here)"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
logs = [Path(p) for p in sys.argv[1:]] or sorted(HERE.glob("*.log"))
rows = []
for log in logs:
    cur = None
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(r"### (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)", line)
        if m:
            cur = {"t": m[1], "log": log.name, "cmd": m[2], "can": [], "exit": None}
            rows.append(cur)
        elif cur is not None and line.startswith("canary"):
            cur["can"].append(line.split()[1])
        elif cur is not None and line.startswith("exit "):
            cur["exit"] = line.split()[1]
for r in sorted(rows, key=lambda r: r["t"]):
    cmd = re.sub(r"^zsh -c (cd spec && )?", "", r["cmd"]).replace("VOLUNDR_LMHEAD8=64 ", "").replace("$PY ", "")
    print(f"{r['t'][11:16]} {r['log']:16s} {' / '.join(r['can']):12s} exit {r['exit']}  {cmd[:110]}")
