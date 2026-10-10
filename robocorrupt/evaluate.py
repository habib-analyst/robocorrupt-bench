"""Evaluation: success rate per (corruption, severity, method)."""
import numpy as np
import torch

from .corruptions import apply
from .models import PolicyCNN, CorrAdapter
from .sim import ReachingEnv
from .train import to_tensor


@torch.no_grad()
def _act(policy, img_u8, adapter=None, device="cpu"):
    """Policy predicts normalized target xy; scale to world coords, IK it."""
    import math
    from .sim import ik
    x = to_tensor(img_u8[None]).to(device)
    if adapter is not None:
        x = adapter(x)
    xy = policy(x).cpu().numpy()[0] * 0.5
    return ik(xy)


def evaluate(policy, n_episodes, seed, corruption=None, severity=0.0,
             adapter=None, device="cpu"):
    """Run episodes; optionally corrupt every observation.

    Returns (success_rate, mean_steps_to_success).
    """
    env = ReachingEnv(seed=seed)
    policy = policy.to(device).eval()
    if adapter is not None:
        adapter = adapter.to(device).eval()
    crng = np.random.default_rng(seed + 999)
    wins, steps = 0, []
    for _ in range(n_episodes):
        obs = env.reset()
        done, ok = False, False
        while not done:
            o = obs
            if corruption is not None:
                o = apply(obs, corruption, severity, crng)
            q_star = _act(policy, o, adapter, device)
            obs, ok, done = env.step_target(q_star)
        wins += int(ok)
        if ok:
            steps.append(env.t)
    env.close()
    return wins / n_episodes, float(np.mean(steps)) if steps else float("nan")


def load_policy(path, device="cpu"):
    m = PolicyCNN()
    m.load_state_dict(torch.load(path, map_location=device))
    return m


def load_adapter(path, device="cpu"):
    m = CorrAdapter()
    m.load_state_dict(torch.load(path, map_location=device))
    return m
