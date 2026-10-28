"""Verhoeff check digit and the Aadhaar-style validity gate (hard gate, not scored evidence)."""
from __future__ import annotations

import numpy as np

_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_check_digit(payload: str) -> int:
    """Check digit to append to `payload` (digits only)."""
    c = 0
    for i, ch in enumerate(reversed(payload)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return _INV[c]


def verhoeff_valid(number: str) -> bool:
    """True if the full string, including its last digit, passes the Verhoeff check."""
    if not number.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def aadhaar_like(rng: np.random.Generator, valid: bool) -> str:
    """Synthetic 12-digit number: first digit 2-9, correct check digit if `valid`, else a wrong one."""
    payload = str(rng.integers(2, 10)) + "".join(str(x) for x in rng.integers(0, 10, size=10))
    check = verhoeff_check_digit(payload)
    if not valid:
        check = (check + int(rng.integers(1, 10))) % 10
    return payload + str(check)


def gate_pass(numbers: list[str]) -> np.ndarray:
    """Hard gate: 12 digits, leading digit not 0/1, Verhoeff-valid."""
    return np.array([len(n) == 12 and n[0] not in "01" and verhoeff_valid(n) for n in numbers], dtype=bool)


def synthetic_gate(rng: np.random.Generator, n: int, invalid_rate: float) -> np.ndarray:
    """Gate outcome for n sessions; invalidity is drawn independently of the class label."""
    flags = rng.random(n) < invalid_rate
    numbers = [aadhaar_like(rng, valid=not f) for f in flags]
    return gate_pass(numbers)
