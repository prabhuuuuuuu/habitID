"""Paths and configuration loading. Raw data is only ever read, never written."""
from __future__ import annotations

from pathlib import Path

import yaml

RESEARCH = Path(__file__).resolve().parents[3]
PROJECT = RESEARCH.parent
RAW = PROJECT / "raw dataset"
DATA = RESEARCH / "data"
INTERIM = DATA / "interim"
PROC = DATA / "processed"
RESULTS = RESEARCH / "results"
CONFIGS = RESEARCH / "configs"
PAPER = RESEARCH / "paper"
PAPER_ASSETS = RESEARCH / "paper_assets"


def load_yaml(name: str) -> dict:
    with open(CONFIGS / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cfg() -> dict:
    return load_yaml("default.yaml")


def load_otp() -> dict:
    return load_yaml("otp_assumptions.yaml")


def load_v0() -> dict:
    return load_yaml("v0_weights.yaml")


def alpha0(cfg: dict, active: list[str]) -> dict[str, float]:
    """Normalized hand-set weights restricted to the active signals (alpha0 = w / 60)."""
    raw = cfg["model"]["alpha0_raw"]
    total = sum(raw.values())
    return {k: raw[k] / total for k in active}
