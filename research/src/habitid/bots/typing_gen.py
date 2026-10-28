"""Typing attacker families. Every generator is fitted on training-fold human typing only.

A naive     fixed or near-fixed intervals with small jitter
B humanized i.i.d. log-normal holds and DDs fitted to pooled human marginals
C replay    perturbed copies of training-subject sessions (multiplicative + time-warp + additive noise)
D learned   autoregressive GRU with log-normal heads, trained on human training sequences
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..features.features import flight


class NaiveGen:
    letter = "A"

    def __init__(self, cfg: dict):
        self.c = cfg["generators"]

    def fit(self, h, d):
        return self

    def sample(self, n: int, rng: np.random.Generator):
        ch = rng.uniform(*self.c["naive_hold_range"], size=(n, 1))
        cd = rng.uniform(*self.c["naive_dd_range"], size=(n, 1))
        j = self.c["naive_jitter_s"]
        h = ch + rng.normal(0, j, size=(n, 10))
        d = cd + rng.normal(0, j, size=(n, 9))
        return np.clip(h, 0.01, None), np.clip(d, 0.01, None)


class HumanizedGen:
    letter = "B"

    def __init__(self, cfg: dict):
        self.c = cfg["generators"]

    def fit(self, h, d):
        lh, ld = np.log(h).ravel(), np.log(d).ravel()
        self.mh, self.sh = lh.mean(), lh.std()
        self.md, self.sd = ld.mean(), ld.std()
        return self

    def sample(self, n: int, rng: np.random.Generator):
        h = np.exp(rng.normal(self.mh, self.sh, size=(n, 10)))
        d = np.exp(rng.normal(self.md, self.sd, size=(n, 9)))
        return h, d


class ReplayGen:
    letter = "C"

    def __init__(self, cfg: dict):
        self.c = cfg["generators"]

    def fit(self, h, d):
        self.h, self.d = h, d
        return self

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self.h), size=n)
        h, d = self.h[idx].copy(), self.d[idx].copy()
        warp = np.exp(rng.normal(0, self.c["replay_warp_logsd"], size=(n, 1)))
        h = h * np.exp(rng.normal(0, self.c["replay_logsd"], size=h.shape)) * warp
        d = d * np.exp(rng.normal(0, self.c["replay_logsd"], size=d.shape)) * warp
        s = self.c["replay_add_s"]
        h = np.clip(h + rng.normal(0, s, size=h.shape), 0.01, None)
        d = np.clip(d + rng.normal(0, s, size=d.shape), 0.01, None)
        return h, d


class _ARNet(nn.Module):
    """Teacher-forced AR model over the 10 key positions. Hold is predicted first, DD conditioned on it."""

    def __init__(self, hidden: int):
        super().__init__()
        self.cell = nn.GRUCell(2 + 10, hidden)
        self.head_h = nn.Linear(hidden, 2)
        self.head_d = nn.Linear(hidden + 1, 2)

    @staticmethod
    def _onehot(t: int, n: int, device) -> torch.Tensor:
        v = torch.zeros(n, 10, device=device)
        v[:, t] = 1.0
        return v

    def _step(self, prev_h, prev_d, t, state):
        x = torch.cat([prev_h[:, None], prev_d[:, None], self._onehot(t, prev_h.shape[0], prev_h.device)], 1)
        state = self.cell(x, state)
        mh, lsh = self.head_h(state).unbind(-1)
        return state, mh, lsh.clamp(-5, 2)

    def _dhead(self, state, zh):
        md, lsd = self.head_d(torch.cat([state, zh[:, None]], 1)).unbind(-1)
        return md, lsd.clamp(-5, 2)

    def nll(self, zh, zd):
        n = zh.shape[0]
        state = torch.zeros(n, self.cell.hidden_size, device=zh.device)
        prev_h = torch.zeros(n, device=zh.device)
        prev_d = torch.zeros(n, device=zh.device)
        total = 0.0
        for t in range(10):
            state, mh, lsh = self._step(prev_h, prev_d, t, state)
            total = total + _gauss_nll(zh[:, t], mh, lsh)
            if t < 9:
                md, lsd = self._dhead(state, zh[:, t])
                total = total + _gauss_nll(zd[:, t], md, lsd)
                prev_d = zd[:, t]
            prev_h = zh[:, t]
        return total.mean()

    @torch.no_grad()
    def sample(self, n: int):
        state = torch.zeros(n, self.cell.hidden_size)
        prev_h = torch.zeros(n)
        prev_d = torch.zeros(n)
        zh_all = torch.zeros(n, 10)
        zd_all = torch.zeros(n, 9)
        for t in range(10):
            state, mh, lsh = self._step(prev_h, prev_d, t, state)
            zh = mh + torch.randn(n) * lsh.exp()
            zh_all[:, t] = zh
            if t < 9:
                md, lsd = self._dhead(state, zh)
                zd = md + torch.randn(n) * lsd.exp()
                zd_all[:, t] = zd
                prev_d = zd
            prev_h = zh
        return zh_all.numpy(), zd_all.numpy()


def _gauss_nll(x, mu, logsd):
    return 0.5 * ((x - mu) * torch.exp(-logsd)) ** 2 + logsd


class ARGen:
    letter = "D"

    def __init__(self, cfg: dict, seed: int):
        self.c = cfg["generators"]
        self.seed = seed

    def fit(self, h, d):
        lh, ld = np.log(h), np.log(d)
        self.mu_h, self.sd_h = lh.mean(0), lh.std(0) + 1e-6
        self.mu_d, self.sd_d = ld.mean(0), ld.std(0) + 1e-6
        zh = torch.tensor((lh - self.mu_h) / self.sd_h, dtype=torch.float32)
        zd = torch.tensor((ld - self.mu_d) / self.sd_d, dtype=torch.float32)
        torch.manual_seed(self.seed)
        self.net = _ARNet(self.c["ar_hidden"])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.c["ar_lr"])
        g = torch.Generator().manual_seed(self.seed)
        bs = self.c["ar_batch"]
        self.train_nll = []
        for _ in range(self.c["ar_epochs"]):
            perm = torch.randperm(len(zh), generator=g)
            tot, nb = 0.0, 0
            for i in range(0, len(zh), bs):
                b = perm[i:i + bs]
                loss = self.net.nll(zh[b], zd[b])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                tot += loss.item(); nb += 1
            self.train_nll.append(tot / nb)
        self.net.eval()
        return self

    def sample(self, n: int, rng=None):
        zh, zd = self.net.sample(n)
        lh = zh * self.sd_h + self.mu_h
        ld = zd * self.sd_d + self.mu_d
        return np.exp(lh), np.exp(ld)

    def state_dict(self):
        return {"net": self.net.state_dict(), "mu_h": self.mu_h, "sd_h": self.sd_h, "mu_d": self.mu_d,
                "sd_d": self.sd_d, "train_nll": self.train_nll}

    def load_state(self, st: dict, cfg: dict):
        self.net = _ARNet(cfg["generators"]["ar_hidden"])
        self.net.load_state_dict(st["net"])
        self.net.eval()
        self.mu_h, self.sd_h, self.mu_d, self.sd_d = st["mu_h"], st["sd_h"], st["mu_d"], st["sd_d"]
        self.train_nll = st["train_nll"]
        return self


def flight_of(h, d):
    return flight(h, d)
