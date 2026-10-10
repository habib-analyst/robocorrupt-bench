"""RoboCorrupt-Bench full experiment driver.

Stages:
  1. collect demos (expert IK rollouts, clean visuals)
  2. train clean BC policy
  3. train augmentation-baseline policy (corruption-augmented demos)
  4. train CorrAdapter against frozen clean policy
  5. evaluate matrix: 10 corruptions x 3 severities x 3 methods x N episodes
  6. clean (no-corruption) reference for every method

Results append incrementally to results/results.jsonl (fsync per line);
--resume skips finished stages and evaluated cells. Writes DONE on completion.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robocorrupt.corruptions import CORRUPTIONS, SEVERITY_LEVELS
from robocorrupt.evaluate import evaluate, load_adapter, load_policy
from robocorrupt.train import collect_demos, train_adapter, train_bc

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _log(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _done_cells(path):
    cells = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    cells.add((r["corruption"], r["severity"], r["method"]))
                except Exception:
                    pass
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=400)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--epochs-bc", type=int, default=15)
    ap.add_argument("--epochs-ad", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stages", type=str, default="all",
                    help="comma list of: demos,bc,aug,adapter,eval (or all)")
    args = ap.parse_args()

    res_dir = os.path.join(BASE, "results")
    os.makedirs(res_dir, exist_ok=True)
    res_path = os.path.join(res_dir, "results.jsonl")
    demos_path = os.path.join(res_dir, "demos.npz")
    bc_dir = os.path.join(res_dir, "policy_clean")
    aug_dir = os.path.join(res_dir, "policy_aug")
    ad_dir = os.path.join(res_dir, "adapter")

    stages = args.stages.split(",") if args.stages != "all" else \
        ["demos", "bc", "aug", "adapter", "eval"]

    if "demos" in stages and not (
            args.resume and os.path.exists(demos_path)):
        n = collect_demos(args.demos, args.seed, demos_path)
        print(f"demos: {n} (obs, action) pairs", flush=True)

    if "bc" in stages and not (
            args.resume and os.path.exists(os.path.join(bc_dir, "best.pt"))):
        b = train_bc(demos_path, bc_dir, epochs=args.epochs_bc,
                     seed=args.seed, resume=args.resume)
        print(f"bc clean: best val {b:.4f}", flush=True)

    if "aug" in stages and not (
            args.resume and os.path.exists(os.path.join(aug_dir, "best.pt"))):
        b = train_bc(demos_path, aug_dir, epochs=args.epochs_bc,
                     augment=True, seed=args.seed + 1, resume=args.resume)
        print(f"bc augment: best val {b:.4f}", flush=True)

    if "adapter" in stages and not (
            args.resume and os.path.exists(os.path.join(ad_dir, "best.pt"))):
        b = train_adapter(demos_path, os.path.join(bc_dir, "best.pt"),
                          ad_dir, epochs=args.epochs_ad,
                          seed=args.seed + 2, resume=args.resume)
        print(f"adapter: best val {b:.4f}", flush=True)

    if "eval" in stages:
        done = _done_cells(res_path) if args.resume else set()
        pol_clean = load_policy(os.path.join(bc_dir, "best.pt"))
        pol_aug = load_policy(os.path.join(aug_dir, "best.pt"))
        adapter = load_adapter(os.path.join(ad_dir, "best.pt"))
        methods = {
            "clean": (pol_clean, None),
            "augment": (pol_aug, None),
            "adapter": (pol_clean, adapter),
        }
        # clean reference first
        for mname, (pol, ad) in methods.items():
            key = ("none", 0.0, mname)
            if key not in done:
                sr, ms = evaluate(pol, args.episodes, args.seed + 100,
                                  adapter=ad)
                _log(res_path, {"corruption": "none", "severity": 0.0,
                                "method": mname, "success": sr,
                                "mean_steps": ms, "n": args.episodes})
                print(f"eval none/{mname}: {sr:.3f}", flush=True)
        for cname in CORRUPTIONS:
            for sev in SEVERITY_LEVELS:
                for mname, (pol, ad) in methods.items():
                    key = (cname, sev, mname)
                    if key in done:
                        continue
                    sr, ms = evaluate(pol, args.episodes,
                                      args.seed + 200, corruption=cname,
                                      severity=sev, adapter=ad)
                    _log(res_path, {"corruption": cname, "severity": sev,
                                    "method": mname, "success": sr,
                                    "mean_steps": ms, "n": args.episodes})
                    print(f"eval {cname}/{sev}/{mname}: {sr:.3f}", flush=True)

    if args.stages == "all":
        open(os.path.join(res_dir, "DONE"), "w").write("done\n")
        print("BENCHMARK DONE", flush=True)


if __name__ == "__main__":
    main()
