import torch
from torch import nn


def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)


class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, bias=False),  # 1, 40, 80 -> 32, 20, 40
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, 2, bias=False),  #  32, 20, 40 -> 64, 10, 20
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, 2, bias=False),  #  64, 10, 20 -> 128, 5, 10
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 256, 3, 2, bias=False),  #  64, 5, 10 -> 128, 3, 5
            nn.LeakyReLU(0.05),
            # nn.Conv2d(128, 128, 3, 2, bias=False),  #  64, 4, 6 -> 128, 2, 4
            # nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(1024, 192, bias=False),
        )
        self.state_proj = nn.Linear(dim_obs, 192)
        self.state_proj.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x: torch.Tensor, v, hx=None):
        img_feat = self.stem(x)
        x = self.act(img_feat + self.state_proj(v))
        hx = self.gru(x, hx)
        act = self.fc(self.act(hx))
        return act, None, hx


if __name__ == "__main__":
    Model()
