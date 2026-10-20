"""CPU-only check of specdec.lookup_replay: replay recorded suite runs group by group into one fresh Lookup per group,
as a resumed ab_suite process would; every step's re-derived proposal must match the recorded draft source.
usage: python check_replay.py <run.jsonl> [<run.jsonl> ...]"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "spec"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import specdec as SD  # noqa: E402
from lookup import Lookup  # noqa: E402

GROUPS = ["coding,humanities,json,math,multilingual,qa,rag", "reasoning,roleplay,stem,summarization,writing",
          "agent,long"]
cfg = json.loads((HERE.parent / "models/volundr-int4-mlx/config.json").read_text())
eos = cfg.get("eos_token_id")
eos = set(eos if isinstance(eos, list) else [eos])
for path in sys.argv[1:]:
    recs = {r["name"]: r for r in map(json.loads, open(path))}
    for g in GROUPS:
        look = Lookup()
        ps = [e for e in SD.suite(HERE / "suite.json", "tune", 0, g.split(",")) if e["name"] in recs]
        steps = 0
        for e in ps:
            SD.lookup_replay(look, e["ids"], recs[e["name"]], eos)
            steps += len(recs[e["name"]]["acc"])
        print(f"{Path(path).name} [{g}]: {len(ps)} prompts, {steps} steps replayed, all proposals match", flush=True)
