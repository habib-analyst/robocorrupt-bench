import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robocorrupt import corruptions as C
from robocorrupt.models import PolicyCNN, CorrAdapter, count_params
from robocorrupt.sim import ReachingEnv, fk, ik
from robocorrupt.train import to_tensor


def test_fk_ik_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-2.5, 2.5, size=2)
        xy = fk(q)
        if np.linalg.norm(xy) > 0.54 or np.linalg.norm(xy) < 0.1:
            continue
        q2 = ik(xy)
        assert np.allclose(fk(q2), xy, atol=1e-6), (q, q2)


def test_sim_reset_step():
    env = ReachingEnv(seed=1)
    obs = env.reset()
    assert obs.shape == (64, 64, 3) and obs.dtype == np.uint8
    a = env.expert_action()
    assert a.shape == (2,)
    obs2, ok, done = env.step(a)
    assert obs2.shape == (64, 64, 3)
    assert isinstance(ok, (bool, np.bool_)) and isinstance(done, (bool, np.bool_))
    # expert should usually succeed from a fresh reset
    env2 = ReachingEnv(seed=2)
    env2.reset()
    for _ in range(60):
        _, ok, done = env2.step(env2.expert_action())
        if done:
            break
    assert ok, "analytic-IK expert failed to reach target"
    env.close()
    env2.close()


def test_corruptions_deterministic_and_severity():
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    for name in C.CORRUPTIONS:
        r1 = np.random.default_rng(123)
        r2 = np.random.default_rng(123)
        o1 = C.apply(img, name, 0.7, r1)
        o2 = C.apply(img, name, 0.7, r2)
        assert o1.shape == img.shape and o1.dtype == np.uint8
        assert np.array_equal(o1, o2), f"{name} not deterministic"
    # severity monotonicity (mean abs diff grows) for a few corruptions
    for name in ["gaussian_noise", "blur", "fog", "pixelate"]:
        diffs = []
        for s in [0.2, 0.6, 1.0]:
            o = C.apply(img, name, s, np.random.default_rng(5))
            diffs.append(np.mean(np.abs(o.astype(int) - img.astype(int))))
        assert diffs[0] < diffs[2], (name, diffs)


def test_model_shapes():
    torch.manual_seed(0)
    pol = PolicyCNN()
    x = torch.rand(4, 3, 64, 64)
    y = pol(x)
    assert y.shape == (4, 2)
    assert y.min() >= -1.0 and y.max() <= 1.0
    ad = CorrAdapter()
    xh = ad(x)
    assert xh.shape == x.shape
    # near-identity at init
    assert torch.mean((xh - x).abs()).item() < 0.05
    assert count_params(pol) > 100_000


def test_to_tensor():
    imgs = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    t = to_tensor(imgs)
    assert t.shape == (2, 3, 64, 64) and t.dtype == torch.float32


def test_train_bc_smoke(tmp_path):
    from robocorrupt.train import collect_demos, train_bc
    dp = str(tmp_path / "demos.npz")
    n = collect_demos(6, seed=3, out_path=dp, noise=0.0)
    assert n > 0
    out = str(tmp_path / "bc")
    best = train_bc(dp, out, epochs=1, batch=8, seed=0)
    assert best < float("inf")
    assert os.path.exists(os.path.join(out, "best.pt"))
