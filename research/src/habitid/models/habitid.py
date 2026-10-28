"""habitID-Net: logit p(x) = b + sum_i alpha_i * g_i(x_i), i in {typing, motion, otp}.

g_typing: 2-layer LSTM over 10 keypress steps -> scalar (standardized by a fixed train-fold scale).
g_motion: 1D-CNN over a 3 s, 6-channel window -> scalar (standardized the same way).
g_otp   : fixed parametric log-likelihood ratio on log latency (assumed parameters, not trained).
alpha   : learnable fusion weights, initialized at alpha0 = (25, 20, 15) / 60 restricted to active heads.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ..bots.otp import components


class TypingEncoder(nn.Module):
    """Post-hoc exploratory option `summary`: per-session summary statistics of the z-normalized inputs
    (mean and SD over positions of the hold, DD and asinh-flight channels) are concatenated to the LSTM's
    final hidden state before the linear head. Default off; the main model does not use it."""

    def __init__(self, in_dim: int = 4, hidden: int = 32, layers: int = 2, summary: bool = False):
        super().__init__()
        self.summary = summary
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden + (6 if summary else 0), 1)

    def forward(self, x):  # (B, 10, 4): [z_hold, z_dd, z_flight, is_last]
        out, _ = self.lstm(x)
        h = out[:, -1]
        if self.summary:
            hold, dd, fl = x[:, :, 0], x[:, :-1, 1], x[:, :-1, 2]
            stats = torch.stack([hold.mean(1), hold.std(1), dd.mean(1), dd.std(1), fl.mean(1), fl.std(1)], dim=1)
            h = torch.cat([h, stats], dim=1)
        return self.head(h).squeeze(-1)


class MotionEncoder(nn.Module):
    def __init__(self, channels=(32, 64, 64, 64), in_ch: int = 6):
        super().__init__()
        layers, c = [], in_ch
        for o in channels:
            layers += [nn.Conv1d(c, o, 5, stride=2, padding=2), nn.ReLU()]
            c = o
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(c, 1)

    def forward(self, x):  # (B, 6, 150)
        return self.head(self.net(x).mean(-1)).squeeze(-1)


def _logpdf_mix(x, logw, mu, sd):
    lp = logw[None] - 0.5 * math.log(2 * math.pi) - torch.log(sd)[None] - 0.5 * ((x[:, None] - mu[None]) / sd[None]) ** 2
    return torch.logsumexp(lp, dim=1)


class HabitIDNet(nn.Module):
    def __init__(self, active: list[str], alpha0: dict[str, float], otp_cfg: dict,
                 lstm_hidden: int = 32, lstm_layers: int = 2, cnn_channels=(32, 64, 64, 64),
                 typing_summary: bool = False):
        super().__init__()
        self.active = list(active)
        self.typ = TypingEncoder(4, lstm_hidden, lstm_layers, summary=typing_summary) if "typing" in active else None
        self.mot = MotionEncoder(tuple(cnn_channels)) if "motion" in active else None
        self.alpha = nn.Parameter(torch.tensor([alpha0[a] for a in active], dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(()))
        self.register_buffer("scale", torch.ones(2))
        self.register_buffer("alpha0", torch.tensor([alpha0[a] for a in active], dtype=torch.float32))
        self.llr_clip = float(otp_cfg["llr_clip"])
        self.log_lo, self.log_hi = math.log(otp_cfg["clip_s"][0]), math.log(otp_cfg["clip_s"][1])
        for side, tag in (("human", "h"), ("automated", "a")):
            w, mu, sg = components(otp_cfg, side)
            self.register_buffer(f"{tag}_logw", torch.tensor(np.log(w), dtype=torch.float32))
            self.register_buffer(f"{tag}_mu", torch.tensor(mu, dtype=torch.float32))
            self.register_buffer(f"{tag}_sd", torch.tensor(sg, dtype=torch.float32))

    def otp_llr(self, logt):
        x = logt.clamp(self.log_lo, self.log_hi)
        llr = _logpdf_mix(x, self.h_logw, self.h_mu, self.h_sd) - _logpdf_mix(x, self.a_logw, self.a_mu, self.a_sd)
        return llr.clamp(-self.llr_clip, self.llr_clip)

    def g_values(self, typ, mot, logt) -> dict[str, torch.Tensor]:
        g = {}
        for a in self.active:
            if a == "typing":
                g[a] = self.typ(typ) / self.scale[0]
            elif a == "motion":
                g[a] = self.mot(mot.transpose(1, 2)) / self.scale[1]
            else:
                g[a] = self.otp_llr(logt)
        return g

    def contributions(self, typ, mot, logt) -> dict[str, torch.Tensor]:
        g = self.g_values(typ, mot, logt)
        return {a: self.alpha[i] * g[a] for i, a in enumerate(self.active)}

    def forward(self, typ, mot, logt):
        c = self.contributions(typ, mot, logt)
        out = self.b.expand(typ.shape[0]) if len(c) == 0 else self.b + sum(c.values())
        return out

    @torch.no_grad()
    def fit_scales(self, typ, mot):
        """Rescale each encoder head ONCE at initialization so that its raw output has unit std on the
        training set (scale stays 1 afterwards). Dividing by a tiny initial std instead made every update
        ~300x too large and saturated the encoders (DEVIATIONS D15)."""
        if self.typ is not None:
            s = self.typ(typ).std().clamp_min(1e-3)
            self.typ.head.weight.div_(s); self.typ.head.bias.div_(s)
        if self.mot is not None:
            s = self.mot(mot.transpose(1, 2)).std().clamp_min(1e-3)
            self.mot.head.weight.div_(s); self.mot.head.bias.div_(s)
        self.scale.fill_(1.0)

    def prior_distance(self) -> float:
        return float(torch.linalg.norm(self.alpha - self.alpha0).item())
