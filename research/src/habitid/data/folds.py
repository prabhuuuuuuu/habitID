"""Subject-grouped folds. Typing and motion folds are built independently; a fold index pairs them.

For fold k: `test` subjects are never used for fitting or calibration; `cal` is carved from the
non-test subjects; `train` is the remainder. Splits are by subject, never by session.
"""
from __future__ import annotations

import numpy as np


def _assign_folds(ids: np.ndarray, strata: np.ndarray, n_folds: int, rng: np.random.Generator) -> dict:
    """Round-robin fold assignment within each stratum (e.g. dataset)."""
    fold_of = {}
    for s in np.unique(strata):
        members = rng.permutation(np.unique(ids[strata == s]))
        for i, sid in enumerate(members):
            fold_of[int(sid)] = i % n_folds
    return fold_of


def make_folds(subjects: np.ndarray, strata: np.ndarray, n_folds: int, seed: int, cal_frac: float) -> list[dict]:
    """Return a list (one per fold) of {'test', 'cal', 'train'} subject-id arrays."""
    rng = np.random.default_rng(seed)
    fold_of = _assign_folds(subjects, strata, n_folds, rng)
    all_ids = np.array(sorted(fold_of))
    stratum_of = {int(i): s for i, s in zip(subjects, strata)}
    folds = []
    for k in range(n_folds):
        test = np.array([i for i in all_ids if fold_of[int(i)] == k])
        rest = np.array([i for i in all_ids if fold_of[int(i)] != k])
        cal = []
        for s in np.unique([stratum_of[int(i)] for i in rest]):
            members = rng.permutation(np.array([i for i in rest if stratum_of[int(i)] == s]))
            n_cal = int(round(cal_frac * len(members)))
            cal += list(members[:n_cal])
        cal = np.array(sorted(cal))
        train = np.array(sorted(set(rest.tolist()) - set(cal.tolist())))
        folds.append({"test": test, "cal": cal, "train": train})
    return folds


def check_disjoint(split: dict) -> None:
    names = ["train", "cal", "test"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(split[a].tolist()) & set(split[b].tolist())
            if overlap:
                raise AssertionError(f"subject leakage between {a} and {b}: {sorted(overlap)[:5]}")


def build_typing_folds(typing_subject: np.ndarray, typing_dataset: np.ndarray, n_folds: int, seed: int,
                       cal_frac: float) -> list[dict]:
    subjects = np.unique(typing_subject)
    strata = np.array([int(typing_dataset[typing_subject == s][0]) for s in subjects])
    return make_folds(subjects, strata, n_folds, seed, cal_frac)


def build_motion_folds(motion_subject: np.ndarray, n_folds: int, seed: int, cal_frac: float) -> list[dict]:
    subjects = np.unique(motion_subject)
    strata = np.zeros(len(subjects), dtype=int)
    return make_folds(subjects, strata, n_folds, seed + 1, cal_frac)
