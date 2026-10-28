"""Motion attacker families M-A..M-D on 3-s windows (6 channels, raw units).

M-A zero/constant (emulator-like), M-B per-channel Gaussian matched to human mean/std,
M-C replay of training-subject windows with noise, M-D causal 1D-conv autoregressive model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ZeroGen:
    letter = "A"

    def __init__(self, cfg: dict):
        self.sd = cfg["generators"]["motion_zero_sd"]

    def fit(self, windows):
        self.C = windows.shape[-1]
        return self

    def sample(self, n: int, T: int, rng: np.random.Generator):
        return rng.normal(0, self.sd, size=(n, T, self.C)).astype(np.float32)


class GaussGen:
    letter = "B"

    def fit(self, windows):
        flat = windows.reshape(-1, windows.shape[-1])
        self.mean = flat.mean(0)
        self.std = flat.std(0) + 1e-6
        return self

    def sample(self, n: int, T: int, rng: np.random.Generator):
        return (self.mean + self.std * rng.normal(size=(n, T, len(self.mean)))).astype(np.float32)


class ReplayMotionGen:
    letter = "C"

    def __init__(self, cfg: dict):
        self.frac = cfg["generators"]["motion_replay_noise_frac"]

    def fit(self, windows):
        self.W = windows
        self.std = windows.reshape(-1, windows.shape[-1]).std(0) + 1e-6
        return self

    def sample(self, n: int, T: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self.W), size=n)
        w = self.W[idx]
        return (w + rng.normal(size=w.shape) * self.std * self.frac).astype(np.float32)


class _CausalConvAR(nn.Module):
    """Linear causal 1D-conv AR(ctx) with Gaussian output and per-channel log-std.

    Linear on purpose: a nonlinear AR net learned a gain > 1 that made 150-step samples diverge
    (see DEVIATIONS D14). A linear AR fitted by maximum likelihood on near-stationary sit/stand
    data is stable and remains a 1D convolutional autoregressive model.
    """

    def __init__(self, ctx: int, ch: int = 6):
        super().__init__()
        self.ctx = ctx
        self.conv = nn.Conv1d(ch, ch, ctx)
        self.logsd = nn.Parameter(torch.zeros(ch))
        self.ch = ch

    def forward(self, ctx_x):  # (B, ch, ctx)
        mu = self.conv(ctx_x).squeeze(-1)
        return mu, self.logsd.expand_as(mu).clamp(-6, 1)


class ConvARGen:
    letter = "D"

    def __init__(self, cfg: dict, seed: int):
        self.c = cfg["generators"]
        self.seed = seed

    def fit(self, windows: np.ndarray):
        flat = windows.reshape(-1, windows.shape[-1])
        self.mean = flat.mean(0)
        self.std = flat.std(0) + 1e-6
        Z = ((windows - self.mean) / self.std).astype(np.float32)
        Z = np.transpose(Z, (0, 2, 1))  # (N, ch, T)
        ctx = self.c["motion_ctx"]
        self.ctx = ctx
        Zp = torch.tensor(np.pad(Z, ((0, 0), (0, 0), (ctx, 0))))
        torch.manual_seed(self.seed)
        self.net = _CausalConvAR(ctx, Z.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.c["motion_ar_lr"])
        T = Z.shape[2]
        g = torch.Generator().manual_seed(self.seed)
        self.train_nll = []
        run = 0.0
        for step in range(self.c["motion_ar_steps"]):
            w = torch.randint(0, len(Z), (self.c["motion_ar_batch"],), generator=g)
            t = torch.randint(0, T, (self.c["motion_ar_batch"],), generator=g)
            cidx = t[:, None] + torch.arange(ctx)[None, :]
            ctx_x = Zp[w[:, None, None], torch.arange(Z.shape[1])[None, :, None], cidx[:, None, :]]
            target = Zp[w, :, t + ctx]
            mu, logsd = self.net(ctx_x)
            nll = (0.5 * ((target - mu) * torch.exp(-logsd)) ** 2 + logsd).mean()
            opt.zero_grad()
            nll.backward()
            opt.step()
            run = 0.98 * run + 0.02 * nll.item() if step else nll.item()
            if step % 250 == 249:
                self.train_nll.append(run)
        self.net.eval()
        return self

    @torch.no_grad()
    def sample(self, n: int, T: int, rng=None):
        ch = self.net.ch
        state = torch.zeros(n, ch, self.ctx)
        out = torch.zeros(n, ch, T)
        for t in range(T):
            mu, logsd = self.net(state)
            x = mu + torch.randn_like(mu) * logsd.exp()
            out[:, :, t] = x
            state = torch.cat([state[:, :, 1:], x[:, :, None]], dim=2)
        z = np.transpose(out.numpy(), (0, 2, 1))
        return (z * self.std + self.mean).astype(np.float32)

    def state_dict(self):
        return {"net": self.net.state_dict(), "mean": self.mean, "std": self.std, "ctx": self.ctx,
                "train_nll": self.train_nll}

    def load_state(self, st: dict, cfg: dict):
        self.net = _CausalConvAR(st["ctx"], 6)
        self.net.load_state_dict(st["net"])
        self.net.eval()
        self.mean, self.std, self.ctx, self.train_nll = st["mean"], st["std"], st["ctx"], st["train_nll"]
        return self
