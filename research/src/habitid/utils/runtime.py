"""Seeding, phase timing and results helpers."""
from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .config import RESULTS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def phase_timer(name: str):
    """Record wall-clock time of a phase in results/timing/phases.json (overwrites the phase entry)."""
    t0 = time.time()
    yield
    dt = time.time() - t0
    path = RESULTS / "timing" / "phases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rec[name] = round(dt, 2)
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"[timing] {name}: {dt:.1f}s", flush=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                    encoding="utf-8")
