"""Summarize results/results.jsonl into markdown tables for the paper."""
import json
import os
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    path = os.path.join(BASE, "results", "results.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = defaultdict(dict)
    for r in rows:
        by[(r["corruption"], r["severity"])][r["method"]] = r["success"]

    corrs = sorted({c for c, _ in by}, key=lambda c: (c != "none", c))
    sevs = sorted({s for _, s in by})

    def get(c, s, m):
        v = by.get((c, s), {}).get(m)
        return f"{v:.3f}" if v is not None else "  -- "

    print("# success rate by corruption / severity / method\n")
    print("| corruption | sev | clean | augment | adapter |")
    print("|---|---|---|---|---|")
    for c in corrs:
        for s in sevs:
            print(f"| {c} | {s} | {get(c,s,'clean')} | {get(c,s,'augment')} | {get(c,s,'adapter')} |")

    # headline: mean success over all corrupted cells per method
    for m in ["clean", "augment", "adapter"]:
        vals = [by[k][m] for k in by if k[0] != "none" and m in by[k]]
        print(f"\nmean corrupted success [{m}]: {sum(vals)/len(vals):.4f} (n={len(vals)})")

    # recovery fraction per corruption at max severity
    print("\n# recovery at severity 1.0: (adapter-clean)/(cleanref-clean)")
    cleanref = by.get(("none", 0.0), {}).get("clean", 1.0)
    for c in corrs:
        if c == "none":
            continue
        d = by.get((c, 1.0), {})
        if all(k in d for k in ("clean", "adapter")):
            denom = cleanref - d["clean"]
            rec = (d["adapter"] - d["clean"]) / denom if denom > 1e-9 else float("nan")
            print(f"{c}: clean={d['clean']:.3f} augment={d.get('augment', float('nan')):.3f} "
                  f"adapter={d['adapter']:.3f} recovery={rec:.2f}")


if __name__ == "__main__":
    main()
