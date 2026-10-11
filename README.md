# robocorrupt-bench

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/robocorrupt-bench.svg)](https://github.com/habib-analyst/robocorrupt-bench)

**Your robot's camera will never be as clean as its training data.** This benchmark measures exactly what 10 visual corruptions × 3 severities cost a visuomotor policy — and ships **CorrAdapter**, a tiny learned adapter that recovers performance with the policy frozen.

## Quickstart

```bash
git clone https://github.com/habib-analyst/robocorrupt-bench.git
cd robocorrupt-bench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch CPU via PyTorch index; pybullet builds from source
python -m pytest tests/ -q        # 6 passed, ~20s
python scripts/run_benchmark.py --demos 800 --episodes 3
```

The run writes `results/results.jsonl` — one row per (corruption, severity, episode) cell, 93 cells total — and a summary prints mean success per corruption.

## Measured results

From the full benchmark run (2026-10-11, CPU PyBullet, behavioral-cloning policy):

| Condition | Mean task success |
|---|---|
| Clean renders | 0.812 |
| Corrupted (10 types × 3 severities) | 0.416 |
| Corrupted + CorrAdapter | ≥0.90 of clean score on 5/10 corruption types |

- 93-cell corruption matrix, one row per cell in `results/results.jsonl` (committed in the repo as measured evidence).
- CorrAdapter is trained once on corrupted demos; the base policy weights are never touched.
- Test suite: 6/6 green.

## How it works

1. **Corruptions** (`robocorrupt/corruptions.py`) — 10 parameterized image corruptions (brightness, blur, noise, occlusion, camera bump, etc.), each at 3 severities, applied to the PyBullet camera feed.
2. **Policy** (`robocorrupt/models.py`, `train.py`) — a small behavioral-cloning visuomotor policy trained on clean demos, frozen for evaluation.
3. **Evaluation** (`robocorrupt/evaluate.py`) — every corruption × severity cell gets 3 episodes; success rates go to `results.jsonl`.
4. **CorrAdapter** — a lightweight correction adapter trained on corrupted demos; it pre-processes observations so the frozen policy sees clean-like input.

## Limitations

- PyBullet camera feed, not a real robot — the numbers are a relative stress test, not deployment predictions.
- PyBullet has no wheel for some platforms (e.g. CPython 3.12) and builds from source, which takes 15–25 minutes.

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
