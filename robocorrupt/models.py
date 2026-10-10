"""Policy and adapter networks."""
import torch
import torch.nn as nn


class PolicyCNN(nn.Module):
    """Visuomotor policy: 64x64 RGB -> target (x, y) in [-0.5, 0.5] (tanh).

    Analytic IK + a low-level tracker convert the predicted target to
    joint motion at rollout. Pure visual target localization.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),      # 15x15
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),     # 6x6
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),     # 4x4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512), nn.ReLU(),
            nn.Linear(512, 2), nn.Tanh(),
        )

    def forward(self, x):
        return self.head(self.conv(x))


class CorrAdapter(nn.Module):
    """Learned visual-correction adapter: corrupted RGB -> corrected RGB.

    Residual form ``out = x + scale * net(x)`` with small init so the adapter
    starts near identity and only learns corrections.
    """

    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        # small init -> near-identity start
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return torch.clamp(x + self.scale * self.net(x), 0, 1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
