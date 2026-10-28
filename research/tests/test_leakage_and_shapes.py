"""Leakage tests (subject-grouped splits in every modality and in every composed split) and shape tests."""
import json

import numpy as np
import pytest
import torch

from habitid.bots.motion_gen import ConvARGen, GaussGen, ReplayMotionGen, ZeroGen, _CausalConvAR
from habitid.bots.otp import llr_numpy, sample_latency
from habitid.bots.typing_gen import ARGen, HumanizedGen, NaiveGen, ReplayGen
from habitid.data.folds import build_motion_folds, build_typing_folds, check_disjoint
from habitid.features.features import MotionNorm, TypingNorm
from habitid.models.habitid import HabitIDNet, MotionEncoder, TypingEncoder
from habitid.utils.config import PROC, RESULTS, alpha0, load_cfg, load_otp

cfg = load_cfg()
otp_cfg = load_otp()


def _typing():
    t = np.load(PROC / "typing_canonical.npz")
    return t["subject"], t["dataset"]


def _motion():
    m = np.load(PROC / "motion_canonical.npz")
    return m["subject"]


def test_typing_folds_are_grouped_and_partition_all_subjects():
    subj, ds = _typing()
    folds = build_typing_folds(subj, ds, 5, cfg["project"]["fold_seed"], cfg["project"]["cal_frac"])
    everyone = set(np.unique(subj).tolist())
    seen = []
    for f in folds:
        check_disjoint(f)
        seen += f["test"].tolist()
    assert sorted(seen) == sorted(everyone)  # every subject tested exactly once


def test_motion_folds_are_grouped_and_partition_all_subjects():
    subj = _motion()
    folds = build_motion_folds(subj, 5, cfg["project"]["fold_seed"], cfg["project"]["cal_frac"])
    seen = []
    for f in folds:
        check_disjoint(f)
        seen += f["test"].tolist()
    assert sorted(seen) == sorted(np.unique(subj).tolist())


def test_check_disjoint_raises_on_overlap():
    with pytest.raises(AssertionError):
        check_disjoint({"train": np.array([1, 2]), "cal": np.array([2]), "test": np.array([3])})


@pytest.mark.parametrize("k", range(5))
def test_composed_fold_has_no_subject_overlap(k):
    path = PROC / "folds" / f"fold{k}.npz"
    if not path.exists():
        pytest.skip("fold corpora not built")
    f = np.load(path)
    for key in ("subject", "motion_subject"):
        sets = {}
        for split in ("train", "cal", "test"):
            s = f[f"{split}_{key}"]
            h = f[f"{split}_label"] == 1
            sets[split] = set(s[h & (s >= 0)].tolist())
        assert not (sets["train"] & sets["cal"]), key
        assert not (sets["train"] & sets["test"]), key
        assert not (sets["cal"] & sets["test"]), key


@pytest.mark.parametrize("k", range(5))
def test_fold_roles_match_fold_partition(k):
    path = PROC / "folds" / f"fold{k}.npz"
    if not path.exists():
        pytest.skip("fold corpora not built")
    f = np.load(path)
    subj, ds = _typing()
    tf = build_typing_folds(subj, ds, 5, cfg["project"]["fold_seed"], cfg["project"]["cal_frac"])[k]
    for split in ("train", "cal", "test"):
        s = f[f"{split}_subject"]
        h = f[f"{split}_label"] == 1
        assert set(s[h].tolist()) <= set(tf[split].tolist())


@pytest.mark.parametrize("k", range(5))
def test_generators_fitted_on_train_role_only(k):
    meta = json.loads((RESULTS / "phase2" / "fold_meta.json").read_text())[k]
    gst = torch.load(PROC / "bots" / f"fold{k}_generators.pt", weights_only=False)
    assert len(gst["replay_pool_h"]) == meta["roles"]["train"]["typing_pool_sessions"]


@pytest.mark.parametrize("k", range(5))
def test_corpus_values_obey_validity_bounds(k):
    path = PROC / "folds" / f"fold{k}.npz"
    if not path.exists():
        pytest.skip("fold corpora not built")
    f = np.load(path)
    b = cfg["typing"]["bounds_s"]
    for split in ("train", "cal", "test"):
        h, d = f[f"{split}_h"], f[f"{split}_d"]
        fl = d - h[:, :-1]
        assert (h >= b["hold"][0]).all() and (h <= b["hold"][1]).all()
        assert (d >= b["dd"][0]).all() and (d <= b["dd"][1]).all()
        assert (fl >= b["flight"][0]).all() and (fl <= b["flight"][1]).all()
        assert np.isfinite(f[f"{split}_windows"]).all()


# ---------------- shapes and generator sanity

def test_typing_norm_shapes_and_padding():
    rng = np.random.default_rng(0)
    h = rng.uniform(0.05, 0.2, size=(64, 10)); d = h[:, :9] + rng.uniform(0.0, 0.3, size=(64, 9))
    tn = TypingNorm(0.05).fit(h, d)
    seq, vec = tn.transform(h, d)
    assert seq.shape == (64, 10, 4) and vec.shape == (64, 28)
    assert np.all(seq[:, -1, 1:3] == 0) and np.all(seq[:, -1, 3] == 1)


def test_motion_norm_shape():
    w = np.random.default_rng(0).normal(size=(32, 150, 6)).astype(np.float32)
    assert MotionNorm().fit(w).transform(w).shape == (32, 150, 6)


def test_encoders_output_scalar_per_session():
    x = torch.randn(7, 10, 4)
    assert TypingEncoder()(x).shape == (7,)
    assert MotionEncoder()(torch.randn(7, 6, 150)).shape == (7,)


@pytest.mark.parametrize("active", [["typing"], ["motion"], ["otp"], ["typing", "motion", "otp"], ["motion", "otp"]])
def test_habitid_forward_shapes_and_alpha_init(active):
    m = HabitIDNet(active, alpha0(cfg, active), otp_cfg)
    out = m(torch.randn(5, 10, 4), torch.randn(5, 150, 6), torch.randn(5))
    assert out.shape == (5,)
    expected = np.array([alpha0(cfg, active)[a] for a in active], dtype=np.float32)
    assert np.allclose(m.alpha.detach().numpy(), expected)


def test_alpha0_is_normalized_prior():
    a = alpha0(cfg, ["typing", "motion", "otp"])
    assert np.isclose(a["typing"], 25 / 60) and np.isclose(a["motion"], 20 / 60) and np.isclose(a["otp"], 15 / 60)


def test_otp_samples_positive_and_clipped():
    rng = np.random.default_rng(0)
    for side in ("human", "automated"):
        s = sample_latency(rng, 5000, otp_cfg, side)
        assert np.all(s > 0) and s.min() >= otp_cfg["clip_s"][0] - 1e-9 and s.max() <= otp_cfg["clip_s"][1] + 1e-9


def test_otp_llr_sign_and_finite():
    t = np.array([0.2, 0.5, 8.0, 40.0])
    llr = llr_numpy(t, otp_cfg)
    assert np.isfinite(llr).all()
    assert llr[0] < 0 and llr[2] > 0  # near-zero favours automated; typical human latency favours human


def test_typing_generators_positive_and_shaped():
    rng = np.random.default_rng(0)
    h = rng.uniform(0.06, 0.15, size=(500, 10)); d = h[:, :9] + rng.uniform(0.05, 0.2, size=(500, 9))
    for gen in (NaiveGen(cfg), HumanizedGen(cfg).fit(h, d), ReplayGen(cfg).fit(h, d)):
        hh, dd = gen.sample(200, rng)
        assert hh.shape == (200, 10) and dd.shape == (200, 9)
        assert np.all(hh > 0) and np.all(dd > 0)
    ar = ARGen(cfg, seed=0)
    ar.fit(h, d)  # tiny smoke fit
    hh, dd = ar.sample(50)
    assert hh.shape == (50, 10) and dd.shape == (50, 9) and np.all(hh > 0) and np.all(dd > 0)


def test_motion_generators_shapes_and_stability():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.05, size=(300, 150, 6)).astype(np.float32)
    assert ZeroGen(cfg).fit(W).sample(10, 150, rng).shape == (10, 150, 6)
    assert GaussGen().fit(W).sample(10, 150, rng).shape == (10, 150, 6)
    assert ReplayMotionGen(cfg).fit(W).sample(10, 150, rng).shape == (10, 150, 6)
    md = ConvARGen(dict(cfg, generators=dict(cfg["generators"], motion_ar_steps=50)), seed=0).fit(W)
    x = md.sample(20, 150)
    assert x.shape == (20, 150, 6) and np.isfinite(x).all()
    assert _CausalConvAR(16)(torch.randn(3, 6, 16))[0].shape == (3, 6)


@pytest.mark.parametrize("k", [0])
def test_fold_arrays_have_expected_shapes(k):
    path = PROC / "folds" / f"fold{k}.npz"
    if not path.exists():
        pytest.skip("fold corpora not built")
    f = np.load(path)
    n = len(f["test_h"])
    assert f["test_d"].shape == (n, 9) and f["test_windows"].shape == (n, 150, 6)
    assert f["test_otp_s"].shape == (n,) and f["test_label"].shape == (n,)
