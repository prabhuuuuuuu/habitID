import numpy as np

from habitid.data.validity import aadhaar_like, gate_pass, verhoeff_check_digit, verhoeff_valid


def test_known_example_wikipedia():
    # Standard worked example: the Verhoeff check digit of "236" is 3, so "2363" is valid.
    assert verhoeff_check_digit("236") == 3
    assert verhoeff_valid("2363")
    assert not verhoeff_valid("2362")


def test_detects_every_single_digit_error():
    rng = np.random.default_rng(0)
    for _ in range(200):
        payload = str(rng.integers(2, 10)) + "".join(str(x) for x in rng.integers(0, 10, size=10))
        num = payload + str(verhoeff_check_digit(payload))
        assert verhoeff_valid(num)
        for pos in range(len(num)):
            for sub in "0123456789":
                if sub == num[pos]:
                    continue
                bad = num[:pos] + sub + num[pos + 1:]
                assert not verhoeff_valid(bad), (num, bad)


def test_detects_every_adjacent_transposition_of_distinct_digits():
    rng = np.random.default_rng(1)
    for _ in range(200):
        payload = str(rng.integers(2, 10)) + "".join(str(x) for x in rng.integers(0, 10, size=10))
        num = payload + str(verhoeff_check_digit(payload))
        for pos in range(len(num) - 1):
            if num[pos] == num[pos + 1]:
                continue
            swapped = num[:pos] + num[pos + 1] + num[pos] + num[pos + 2:]
            assert not verhoeff_valid(swapped), (num, swapped)


def test_synthetic_numbers_pass_gate_only_when_valid():
    rng = np.random.default_rng(2)
    valid = [aadhaar_like(rng, True) for _ in range(300)]
    invalid = [aadhaar_like(rng, False) for _ in range(300)]
    assert gate_pass(valid).all()
    assert not gate_pass(invalid).any()


def test_gate_rejects_leading_zero_or_one_and_wrong_length():
    assert not gate_pass(["012345678901"])[0]
    assert not gate_pass(["1" + "2345678901" + "2"])[0]
    assert not gate_pass(["2345"])[0]
