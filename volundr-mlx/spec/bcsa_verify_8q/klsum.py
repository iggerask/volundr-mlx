import json
import sys

for f in sys.argv[1:]:
    print("==", f)
    for line in open(f):
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if "name" in r:
            print(f"{r['name'][:30]:30s} P={r['prompt_tokens']:5d} rows={r['rows']:3d} agree={r['agree']:3d} "
                  f"kl_mean={r['kl_mean']:.2e} p99={r['kl_p99']:.2e} max={r['kl_max']:.2e}")
        else:
            print("ALL", {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in r.items()})
