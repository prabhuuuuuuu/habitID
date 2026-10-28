import numpy as np

from habitid.conformal.conformal import class_thresholds, prediction_sets, verdicts
from habitid.eval.baselines import v0_score
from habitid.utils.config import load_v0


def _sessions(h_val, d_val, n=1):
    h = np.full((n, 10), h_val, dtype=float)
    d = np.full((n, 9), d_val, dtype=float)
    return h, d


def test_v0_hand_computed_review_example():
    # Constant flights -> CV 0 -> s_cv 0; WPM = 120 / (9*0.2) = 66.7 -> s_wpm 1 -> s_typing 0.5.
    # Constant acceleration -> CV 0 -> s_motion 0. OTP 10 s -> s_otp 1. Gate passed -> aadhaar 1.
    # Constant sim/phone terms: 10*0.5 + 10*1 = 15.  S0 = 12.5 + 0 + 20 + 15 + 15 = 62.5 -> REVIEW.
    h, d = _sessions(0.1, 0.2)
    win = np.zeros((1, 150, 6)); win[:, :, 0] = 1.0
    S0, v = v0_score(h, d, win, np.array([10.0]), np.array([True]), load_v0())
    assert np.isclose(S0[0], 62.5)
    assert v[0] == "REVIEW"


def test_v0_all_signals_perfect_is_verified():
    # Flight CV = 0.5 (flights 0.05 / 0.15 around mean 0.1); WPM ~ 66; motion CV = 0.5; OTP 10 s; gate passed.
    # S0 = 25 + 20 + 20 + 15 + 5 + 10 = 95 -> VERIFIED.
    h = np.full((1, 10), 0.1)
    f = np.array([[0.05, 0.15] * 4 + [0.05]])
    d = h[:, :9] + f
    win = np.zeros((1, 150, 6))
    win[0, :, 0] = np.where(np.arange(150) % 2 == 0, 0.5, 1.5)
    S0, v = v0_score(h, d, win, np.array([10.0]), np.array([True]), load_v0())
    assert np.isclose(S0[0], 95.0, atol=1e-6)
    assert v[0] == "VERIFIED"


def test_v0_bot_like_session_is_suspicious():
    # OTP 0.5 s (s_otp 0), constant flights (s_cv 0), zero acceleration (s_motion 0), WPM 12 (s_wpm 0),
    # gate failed (aadhaar 0). S0 = 10*0.5 (sim-age) + 10*1 (phone) = 15 -> SUSPICIOUS.
    h, d = _sessions(0.1, 10.0 / 9)
    win = np.zeros((1, 150, 6))
    S0, v = v0_score(h, d, win, np.array([0.5]), np.array([False]), load_v0())
    assert S0[0] < 45 and v[0] == "SUSPICIOUS"


def test_class_conditional_conformal_marginal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(3)
    eps_list = [0.05, 0.1, 0.2]
    n_cal, n_test = 4000, 40000

    def draw(n):
        y = rng.integers(0, 2, size=n)  # 1 = human
        logit = np.where(y == 1, 1.5, -1.5) + rng.normal(0, 1.2, size=n)
        return y, 1 / (1 + np.exp(-logit))

    y_c, p_c = draw(n_cal)
    y_t, p_t = draw(n_test)
    for eps in eps_list:
        q = class_thresholds(p_c, y_c, eps)
        in_h, in_b = prediction_sets(p_t, q)
        cov_h = in_h[y_t == 1].mean()
        cov_b = in_b[y_t == 0].mean()
        assert cov_h >= 1 - eps - 0.01, (eps, cov_h)
        assert cov_b >= 1 - eps - 0.01, (eps, cov_b)
        assert cov_h <= 1 - eps + 0.02 and cov_b <= 1 - eps + 0.02


def test_conformal_verdict_mapping():
    in_h = np.array([True, False, True, False])
    in_b = np.array([False, True, True, False])
    v = verdicts(in_h, in_b)
    assert list(v) == ["VERIFIED", "SUSPICIOUS", "REVIEW", "REVIEW"]


def test_tiny_calibration_class_gives_infinite_threshold():
    q = class_thresholds(np.array([0.2, 0.8]), np.array([1, 0]), eps=0.01)
    assert q[0] == np.inf or q[1] == np.inf
