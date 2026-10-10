"""Data collection + training: behavior cloning, augmentation baseline, adapter.

All long steps checkpoint per epoch and support --resume via run_benchmark.py.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .corruptions import CORRUPTIONS, apply
from .models import PolicyCNN, CorrAdapter
from .sim import ReachingEnv, ik


def to_tensor(imgs):
    """uint8 NHWC -> float32 NCHW in [0, 1]."""
    t = torch.from_numpy(np.asarray(imgs)).float() / 255.0
    return t.permute(0, 3, 1, 2).contiguous()


def collect_demos(n_demos, seed, out_path, noise=0.0):
    """Roll out the analytic-IK expert; save (obs, target_xy/0.5) pairs.

    The policy predicts the *target position* (normalized to [-1, 1]);
    analytic IK + a low-level tracker convert it to joint motion.
    Target localization from pixels is far more precise than joint-angle
    regression (no IK-branch discontinuities in the label space).
    """
    env = ReachingEnv(seed=seed)
    obs_list, act_list = [], []
    done_count = 0
    while done_count < n_demos:
        obs = env.reset()
        done = False
        while not done:
            q_star = ik(env.target)
            if noise > 0:
                q_star = q_star + env.rng.normal(0, noise, size=2)
            obs_list.append(obs)
            act_list.append((env.target / 0.5).astype(np.float32))
            obs, success, done = env.step_target(q_star)
        done_count += 1
    env.close()
    obs_arr = np.stack(obs_list).astype(np.uint8)
    act_arr = np.stack(act_list).astype(np.float32)
    np.savez_compressed(out_path, obs=obs_arr, act=act_arr)
    return len(obs_list)


def _save_ckpt(path, epoch, model, opt, best, log):
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "opt": opt.state_dict(), "best": best, "log": log}, path)


def _load_ckpt(path, model, opt):
    s = torch.load(path, map_location="cpu")
    model.load_state_dict(s["model"])
    opt.load_state_dict(s["opt"])
    return s["epoch"] + 1, s["best"], s["log"]


def _corrupt_batch(imgs_nhwc, rng):
    """Apply a random corruption/severity to each image in a uint8 batch."""
    names = list(CORRUPTIONS.keys())
    out = np.empty_like(imgs_nhwc)
    for i in range(len(imgs_nhwc)):
        name = names[rng.integers(len(names))]
        sev = float(rng.uniform(0.2, 1.0))
        out[i] = apply(imgs_nhwc[i], name, sev, rng)
    return out


def train_bc(demos_path, out_dir, epochs=15, batch=128, lr=3e-4,
             augment=False, seed=0, resume=False):
    """Behavior cloning. augment=True -> corruption-augmented baseline."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = np.load(demos_path)
    obs_u8, act = d["obs"], d["act"]
    n = len(obs_u8)
    idx = rng.permutation(n)
    nv = max(1, min(n // 10, n - 1))
    tr, va = idx[nv:], idx[:nv]

    def loader(idxs, aug):
        o = obs_u8[idxs]
        if aug:
            o = _corrupt_batch(o, np.random.default_rng(seed + 1))
        ds = TensorDataset(to_tensor(o), torch.from_numpy(act[idxs]))
        return DataLoader(ds, batch_size=batch, shuffle=True)

    tr_dl, va_dl = loader(tr, augment), loader(va, False)
    model = PolicyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "last.pt")
    start, best, log = 0, float("inf"), []
    if resume and os.path.exists(ckpt):
        start, best, log = _load_ckpt(ckpt, model, opt)
    for ep in range(start, epochs):
        model.train()
        tl = 0.0
        for x, y in tr_dl:
            opt.zero_grad()
            loss = mse(model(x), y)
            loss.backward()
            opt.step()
            tl += loss.item()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in va_dl:
                vl += mse(model(x), y).item()
        tl /= len(tr_dl)
        vl /= len(va_dl)
        log.append({"epoch": ep, "train": tl, "val": vl})
        if vl < best:
            best = vl
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        _save_ckpt(ckpt, ep, model, opt, best, log)
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f)
    return best


def train_adapter(demos_path, policy_path, out_dir, epochs=20, batch=128,
                  lr=1e-3, lam=0.5, seed=0, resume=False):
    """Train CorrAdapter against a FROZEN policy.

    Loss = MSE(pi(adapter(corrupted)), expert_action)
           + lam * MSE(adapter(corrupted), clean_obs).
    The first term is the action-consistency objective: the adapter is
    trained so the frozen policy acts correctly under corruption.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 7)
    d = np.load(demos_path)
    obs_u8, act = d["obs"], d["act"]
    policy = PolicyCNN()
    policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
    policy.eval()
    for p_ in policy.parameters():
        p_.requires_grad = False

    adapter = CorrAdapter()
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    mse = nn.MSELoss()
    os.makedirs(out_dir, exist_ok=True)
    n = len(obs_u8)
    idx = rng.permutation(n)
    nv = max(1, min(n // 10, n - 1))
    tr, va = idx[nv:], idx[:nv]
    ckpt = os.path.join(out_dir, "last.pt")
    start, best, log = 0, float("inf"), []
    if resume and os.path.exists(ckpt):
        start, best, log = _load_ckpt(ckpt, adapter, opt)

    def batch_loss(idxs):
        clean_u8 = obs_u8[idxs]
        corr_u8 = _corrupt_batch(clean_u8, rng)
        xc = to_tensor(clean_u8)
        xn = to_tensor(corr_u8)
        y = torch.from_numpy(act[idxs])
        xhat = adapter(xn)
        with torch.no_grad():
            pass
        act_loss = mse(policy(xhat), y)
        rec_loss = mse(xhat, xc)
        return act_loss + lam * rec_loss, act_loss.item(), rec_loss.item()

    for ep in range(start, epochs):
        adapter.train()
        tl = al = rl = 0.0
        nb = 0
        for i in range(0, len(tr), batch):
            opt.zero_grad()
            loss, a_l, r_l = batch_loss(tr[i:i + batch])
            loss.backward()
            opt.step()
            tl += loss.item()
            al += a_l
            rl += r_l
            nb += 1
        adapter.eval()
        vl = 0.0
        with torch.no_grad():
            for i in range(0, len(va), batch):
                loss, _, _ = batch_loss(va[i:i + batch])
                vl += loss.item()
        log.append({"epoch": ep, "train": tl / nb, "val": vl / max(1, len(va) // batch),
                    "act": al / nb, "rec": rl / nb})
        if log[-1]["val"] < best:
            best = log[-1]["val"]
            torch.save(adapter.state_dict(), os.path.join(out_dir, "best.pt"))
        _save_ckpt(ckpt, ep, adapter, opt, best, log)
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f)
    return best
