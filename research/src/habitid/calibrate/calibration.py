"""Temperature scaling (fitted on the calibration split only) and probability helpers."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(logit: np.ndarray, y_human: np.ndarray) -> float:
    """Minimize BCE of sigmoid(logit / T) over T in exp([-3, 3])."""
    z = torch.tensor(logit, dtype=torch.float64)
    yt = torch.tensor(y_human, dtype=torch.float64)

    def nll(log_t):
        return float(F.binary_cross_entropy_with_logits(z / float(np.exp(log_t)), yt).item())

    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def p_human(logit: np.ndarray, T: float = 1.0) -> np.ndarray:
    return sigmoid(logit / T)
