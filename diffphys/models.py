"""Profile-specific CNNs with the original 192-dimensional GRU head."""

import torch
from torch import nn
from torch.nn import functional as F

from .profiles import get_profile


class VanillaModel(nn.Module):
    def __init__(self, dim_obs=10, dim_action=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128 * 2 * 4, 192, bias=False),
        )
        self.state_proj = nn.Linear(dim_obs, 192)
        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.act = nn.LeakyReLU(0.05)
        with torch.no_grad():
            self.state_proj.weight.mul_(0.5)
            self.fc.weight.mul_(0.01)

    def reset(self):
        pass  # Hidden state belongs to each rollout.

    def forward(self, x, state, hx=None):
        features = self.act(self.stem(x) + self.state_proj(state))
        hx = self.gru(features, hx)
        return self.fc(self.act(hx)), None, hx


class ModifiedModel(VanillaModel):
    def __init__(self, dim_obs=10, dim_action=6):
        super().__init__(dim_obs, dim_action)
        # A: 32 -> 16 -> 8 -> 8; retain three Conv/LeakyReLU stages.
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, 2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 192, bias=False),
        )

    def forward(self, x, state, hx=None):
        return super().forward(F.adaptive_max_pool2d(x, (32, 32)), state, hx)


def build_model(profile):
    spec = get_profile(profile) if isinstance(profile, str) else profile
    return (VanillaModel if spec.name == "vanilla" else ModifiedModel)(spec.state_dim, spec.action_dim)
