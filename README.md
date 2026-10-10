# RoboCorrupt-Bench 🤖🌪️

**Your robot's camera will never be as clean as its training data. This benchmark
measures exactly how much that costs — and ships a fix.**

Vision-based robot policies are trained on clean renders and deployed into dirty
worlds: bad lighting, occlusions, sensor noise, camera bumps. RoboCorrupt-Bench
asks the ImageNet-C question for robot control: *how much does each visual
corruption hurt, and can we fix it without retraining the policy?*

## 30-second why

- **Problem**: A visuomotor policy that scores 90% in sim can collapse when the
  camera feed degrades — and nobody had a standard way to measure it.
- **What this is**: 10 parameterized visual corruptions × 3 severities, a
  reproducible CPU-only PyBullet benchmark, and **CorrAdapter** — a tiny learned
  adapter that restores performance with the policy *frozen*.
- **Result**: [headline numbers — see Results]

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt   # torch CPU via PyTorch index; pybullet builds from source
python -m pytest tests/ -q        # 6 tests, ~20s
python scripts/run_benchmark.py --demos 800 --episodes 32 --resume
# results land in results/results.jsonl (one row per cell), DONE on completion
python scripts/summarize.py       # markdown tables for the paper
```

## What's inside

| Module | What it does |
|---|---|
| `robocorrupt/sim.py` | PyBullet 2-link planar reaching, top-down 64×64 camera, analytic-IK expert |
| `robocorrupt/corruptions.py` | 10 seeded corruptions (brightness, contrast, noise, blur, occlusion, hue, jitter, pixelation, fog, speckle) |
| `robocorrupt/models.py` | CNN policy (target localization) + CorrAdapter (residual correction) |
| `robocorrupt/train.py` | Behavior cloning, augmentation baseline, adapter training (all checkpointed) |
| `robocorrupt/evaluate.py` | Success-rate eval per (corruption, severity, method) |
| `scripts/run_benchmark.py` | Full driver with `--resume` (per-cell JSONL, survives restarts) |

## Methods compared

- **`clean`**: CNN policy by behavior cloning on clean demos. No defense.
- **`augment`**: same policy trained on corruption-augmented demos (domain randomization).
- **`adapter`** (novel): CorrAdapter pre-processes observations; trained against the
  **frozen** policy with an **action-consistency objective** — it learns whatever
  correction makes the frozen policy act correctly, plus a small reconstruction term.

## Results

*Measured on 2 CPU cores, $0. 32 episodes/cell, 93-cell matrix (10 corruptions ×
3 severities × 3 methods + clean refs).*

- **Clean policy**: 81.2% success on clean observations — but full-severity
  contrast drops it to 0%, and brightness/occlusion/speckle/hue-shift to
  3–6%. Mean corrupted success: 41.6%.
- **CorrAdapter** (frozen policy + learned correction): recovers ≥90% of lost
  performance on 5/10 corruptions (occlusion 3%→91%, speckle 3%→91%,
  gaussian noise 16%→84%, brightness 6%→75%, camera jitter 66%→81%).
  Mean corrupted success 73.4% — within 3 points of the domain-randomization
  baseline (76.0%) **without retraining the policy**.
- **Honest negative**: blur at severity 1.0 — the adapter (21.9%) underperforms
  even the undefended policy (50.0%). It recovers masked signal; it cannot
  reconstruct destroyed information.

Full table + severity curves: see `papers/PAPER.md` §4 (research workspace).

## Paper

Full writeup: `papers/PAPER.md` in the research workspace (this repo is the code).

## License

MIT — see LICENSE.
