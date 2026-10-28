"""Training of habitID-Net variants: BCE + lambda_prior ||alpha - alpha0||^2 + lambda_adv L_adv.

L_adv: PGD-perturbed bot sessions (feature-space, projected to human quantile bounds) are added to
the batch and the detector is trained to still flag them. Generator-D bot sessions are refreshed
every `refresh_every` epochs by best-of-N selection against the current detector (a discrete
approximation of "refresh G to fool D"; see DEVIATIONS D13).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

from ..bots.policy import LETTER_CODE, valid_typing
from ..features.features import MotionNorm, TypingNorm
from ..models.habitid import HabitIDNet
from ..utils.config import alpha0 as alpha0_fn
from ..utils.runtime import set_seed

LETTER_D = LETTER_CODE["D"]


def tensors_from_split(fold: dict, split: str, tnorm: TypingNorm, mnorm: MotionNorm, use_flight: bool,
                       keep: np.ndarray | None = None) -> dict:
    h, d = fold[f"{split}_h"], fold[f"{split}_d"]
    if keep is not None:
        sel = keep
    else:
        sel = np.ones(len(h), dtype=bool)
    h, d = h[sel], d[sel]
    seq, _ = tnorm.transform(h, d, use_flight)
    out = {
        "typ": torch.tensor(seq), "mot": torch.tensor(mnorm.transform(fold[f"{split}_windows"][sel])),
        "logt": torch.tensor(np.log(fold[f"{split}_otp_s"][sel]), dtype=torch.float32),
        "y": torch.tensor(fold[f"{split}_label"][sel], dtype=torch.float32),
        "letter": fold[f"{split}_letter"][sel], "type": fold[f"{split}_type"][sel],
        "cluster": fold[f"{split}_cluster"][sel], "gate": fold[f"{split}_gate"][sel],
        "subject": fold[f"{split}_subject"][sel], "dataset": fold[f"{split}_dataset"][sel],
        "otp_s": fold[f"{split}_otp_s"][sel], "h": h, "d": d,
    }
    return out


def _bce(logit, y, reduction="mean"):
    return F.binary_cross_entropy_with_logits(logit, y, reduction=reduction)


def _bounds(tr: dict, quant=(0.005, 0.995)):
    human = tr["y"] == 1
    typ = tr["typ"][human]
    mot = tr["mot"][human]
    logt = tr["logt"][human]
    lo_t = torch.quantile(typ, quant[0], dim=0, keepdim=True)
    hi_t = torch.quantile(typ, quant[1], dim=0, keepdim=True)
    lo_m = torch.quantile(mot.reshape(-1, mot.shape[-1]), quant[0], dim=0).view(1, 1, -1)
    hi_m = torch.quantile(mot.reshape(-1, mot.shape[-1]), quant[1], dim=0).view(1, 1, -1)
    lo_l = torch.quantile(logt, quant[0]).view(1)
    hi_l = torch.quantile(logt, quant[1]).view(1)
    return (lo_t, hi_t, lo_m, hi_m, lo_l, hi_l)


def pgd_bots(model, typ, mot, logt, bnd, eps, steps, step):
    """Attacker maximizes the human logit of bot sessions within the human feasibility box.

    eps: float or (eps_typing, eps_motion, eps_otp). Per-modality radii keep the 900-dim motion window
    from dominating the adversarial term (see DEVIATIONS D16)."""
    lo_t, hi_t, lo_m, hi_m, lo_l, hi_l = bnd
    if isinstance(eps, (tuple, list)):
        e_t, e_m, e_l = eps
    else:
        e_t = e_m = e_l = eps
    mask = torch.zeros_like(typ)
    mask[..., :3] = 1.0
    # Projection box: the human feasibility box, widened to contain the original sample. Clipping to the
    # human box alone would move out-of-range bot sessions (e.g. naive bots) onto the boundary (DEVIATIONS D16).
    lo_t, hi_t = torch.minimum(lo_t, typ), torch.maximum(hi_t, typ)
    lo_m, hi_m = torch.minimum(lo_m, mot), torch.maximum(hi_m, mot)
    lo_l, hi_l = torch.minimum(lo_l, logt), torch.maximum(hi_l, logt)
    dt = (torch.rand_like(typ) * 2 - 1) * e_t * mask
    dm = (torch.rand_like(mot) * 2 - 1) * e_m
    dl = (torch.rand_like(logt) * 2 - 1) * e_l
    zeros = torch.zeros(typ.shape[0])
    for _ in range(steps):
        xt = (typ + dt).clamp(lo_t, hi_t).detach().requires_grad_(True)
        xm = (mot + dm).clamp(lo_m, hi_m).detach().requires_grad_(True)
        xl = (logt + dl).clamp(lo_l, hi_l).detach().requires_grad_(True)
        loss = F.binary_cross_entropy_with_logits(model(xt, xm, xl), zeros, reduction="sum")
        gt, gm, gl = torch.autograd.grad(loss, [xt, xm, xl], allow_unused=True)
        gt = torch.zeros_like(xt) if gt is None else gt
        gm = torch.zeros_like(xm) if gm is None else gm
        gl = torch.zeros_like(xl) if gl is None else gl
        dt = ((dt + step * gt.sign()).clamp(-e_t, e_t)) * mask
        dm = (dm + step * gm.sign()).clamp(-e_m, e_m)
        dl = (dl + step * gl.sign()).clamp(-e_l, e_l)
    return ((typ + dt).clamp(lo_t, hi_t).detach(), (mot + dm).clamp(lo_m, hi_m).detach(),
            (logt + dl).clamp(lo_l, hi_l).detach())


@torch.no_grad()
def refresh_D(model, tr: dict, idx: np.ndarray, arg, tnorm: TypingNorm, bounds_s: dict, n_best: int,
              use_flight: bool) -> None:
    M = len(idx)
    h_c, d_c = arg.sample(M * n_best)
    valid = torch.tensor(valid_typing(h_c, d_c, bounds_s)).view(M, n_best)
    seq, _ = tnorm.transform(h_c, d_c, use_flight)
    seq = torch.tensor(seq)
    idx_t = torch.tensor(idx)
    mot = tr["mot"][idx_t].repeat_interleave(n_best, 0)
    logt = tr["logt"][idx_t].repeat_interleave(n_best, 0)
    logit = model(seq, mot, logt).view(M, n_best)
    logit = logit.masked_fill(~valid, float("-inf"))
    best = logit.argmax(1)
    flat = torch.arange(M) * n_best + best
    tr["typ"][idx_t] = seq[flat]


@torch.no_grad()
def predict(model, t: dict, bs: int = 4096) -> dict:
    model.eval()
    logit, comp = [], {a: [] for a in ("typing", "motion", "otp")}
    for i in range(0, len(t["typ"]), bs):
        c = model.contributions(t["typ"][i:i + bs], t["mot"][i:i + bs], t["logt"][i:i + bs])
        total = model.b + sum(c.values())
        logit.append(total.numpy())
        for a in comp:
            comp[a].append(c[a].numpy() if a in c else np.zeros(len(total)))
    out = {"logit": np.concatenate(logit)}
    for a in comp:
        out[f"contrib_{a}"] = np.concatenate(comp[a])
    return out


def fit_temperature(logit: np.ndarray, y: np.ndarray) -> float:
    z = torch.tensor(logit, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)

    def nll(log_t):
        return float(_bce(z / float(np.exp(log_t)), yt).item())

    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def train_variant(fold: dict, cfg: dict, otp_cfg: dict, gen_d, variant: dict, seed: int) -> dict:
    """Train one variant on one fold with one seed. Returns {'model', 'preds', 'meta', 'tnorm', 'mnorm'}.

    gen_d: fitted ARGen used for the generator-D refresh and for the adaptive D* test attack.
    """
    set_seed(seed)
    t0 = time.time()
    tcfg = cfg["training"]
    active = variant["active"]
    use_flight = variant.get("use_flight", True)
    exclude = [LETTER_CODE[x] for x in variant.get("exclude", [])]

    train_human = fold["train_label"] == 1
    tnorm = TypingNorm(cfg["typing"]["flight_asinh_scale"]).fit(fold["train_h"][train_human],
                                                             fold["train_d"][train_human])
    mnorm = MotionNorm().fit(fold["train_motion_pool"])

    keep_tr = ~np.isin(fold["train_letter"], exclude)
    keep_cal = ~np.isin(fold["cal_letter"], exclude)
    tr = tensors_from_split(fold, "train", tnorm, mnorm, use_flight, keep_tr)
    cal = tensors_from_split(fold, "cal", tnorm, mnorm, use_flight, keep_cal)
    te = tensors_from_split(fold, "test", tnorm, mnorm, use_flight)

    alpha0 = alpha0_fn(cfg, active)
    model = HabitIDNet(active, alpha0, otp_cfg, cfg["model"]["lstm_hidden"], cfg["model"]["lstm_layers"],
                       cfg["model"]["cnn_channels"], typing_summary=bool(variant.get("typing_summary", False)))
    model.fit_scales(tr["typ"], tr["mot"])
    bnd = _bounds(tr, tuple(cfg["motion"]["quantile_bounds"]))
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])

    lam_p = float(variant.get("lam_prior", tcfg["lambda_prior"]))
    lam_a = float(variant.get("lam_adv", tcfg["lambda_adv"]))
    adv = lam_a > 0 and any(a in active for a in ("typing", "motion", "otp"))
    refresh_ok = (adv and "typing" in active and (LETTER_D not in exclude) and gen_d is not None
                  and bool((tr["letter"] == LETTER_D).any()))
    d_idx = np.where(tr["letter"] == LETTER_D)[0] if refresh_ok else None
    rng = np.random.default_rng(seed)

    n = len(tr["y"])
    best = {"loss": float("inf"), "state": None, "epoch": -1}
    bad = 0
    history = []
    a0 = torch.tensor([alpha0[a] for a in active], dtype=torch.float32)
    for ep in range(tcfg["epochs"]):
        te0 = time.time()
        if refresh_ok and ep > 0 and ep % tcfg["refresh_every"] == 0:
            refresh_D(model, tr, d_idx, gen_d, tnorm, cfg["typing"]["bounds_s"],
                      tcfg["best_of_n"], use_flight)
        model.train()
        perm = torch.randperm(n)
        run_loss, nb = 0.0, 0
        for i in range(0, n, tcfg["batch_size"]):
            b = perm[i:i + tcfg["batch_size"]]
            typ, mot, logt, y = tr["typ"][b], tr["mot"][b], tr["logt"][b], tr["y"][b]
            loss = _bce(model(typ, mot, logt), y)
            if lam_p > 0:
                loss = loss + lam_p * ((model.alpha - a0) ** 2).sum()
            if adv:
                bot = y == 0
                if bot.any():
                    eps = (tcfg["pgd_eps"], tcfg["pgd_eps_motion"], tcfg["pgd_eps_otp"])
                    xt, xm, xl = pgd_bots(model, typ[bot], mot[bot], logt[bot], bnd, eps,
                                          tcfg["pgd_steps"], tcfg["pgd_step"])
                    l_adv = _bce(model(xt, xm, xl), torch.zeros(int(bot.sum())))
                    loss = loss + lam_a * l_adv
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += float(loss.item()); nb += 1
        model.eval()
        with torch.no_grad():
            cal_logit = model(cal["typ"], cal["mot"], cal["logt"])
            cal_loss = float(_bce(cal_logit, cal["y"]).item())
        history.append({"epoch": ep, "train_loss": run_loss / max(nb, 1), "cal_bce": cal_loss,
                        "alpha": model.alpha.detach().tolist(), "b": float(model.b.item()),
                        "epoch_s": round(time.time() - te0, 2), "refreshed_D": bool(refresh_ok and ep > 0 and ep % tcfg["refresh_every"] == 0)})
        if cal_loss < best["loss"] - 1e-4:
            best = {"loss": cal_loss, "state": copy.deepcopy(model.state_dict()), "epoch": ep}
            bad = 0
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    model.load_state_dict(best["state"])
    model.eval()

    pc = predict(model, cal)
    T = fit_temperature(pc["logit"], cal["y"].numpy())
    preds = {}
    for name, t in (("cal", cal), ("test", te)):
        p = predict(model, t)
        for k, v in p.items():
            preds[f"{name}_{k}"] = v
        for k in ("y", "letter", "type", "cluster", "gate", "otp_s", "dataset", "subject"):
            arr = t[k].numpy() if torch.is_tensor(t[k]) else np.asarray(t[k])
            preds[f"{name}_{k}"] = arr
    preds["T"] = np.array(T)
    preds["alpha"] = model.alpha.detach().numpy()
    preds["alpha0"] = a0.numpy()
    if gen_d is not None and variant.get("dstar", False):
        star = make_dstar(model, te, gen_d, tnorm, cfg, use_flight, fold["fold_index"])
        for k, v in star.items():
            preds[f"star_{k}"] = v
    meta = {
        "active": active, "lam_prior": lam_p, "lam_adv": lam_a, "exclude": variant.get("exclude", []),
        "use_flight": use_flight, "alpha": model.alpha.detach().tolist(), "alpha0": a0.tolist(),
        "prior_distance": model.prior_distance(), "T": T, "best_epoch": best["epoch"],
        "epochs_run": len(history), "history": history, "n_train": int(n), "n_cal": int(len(cal["y"])),
        "n_test": int(len(te["y"])), "seed": seed, "runtime_s": round(time.time() - t0, 2),
        "adv_active": bool(adv), "generatorD_refresh": bool(refresh_ok),
        "otp_scale_fixed": True, "scale": model.scale.tolist(),
    }
    return {"model": model, "preds": preds, "meta": meta, "tnorm": tnorm, "mnorm": mnorm}


@torch.no_grad()
def make_dstar(model, te: dict, arg, tnorm: TypingNorm, cfg: dict, use_flight: bool, fold_index: int) -> dict:
    """Adaptive attacker: D-family full-type test sessions, typing re-selected best-of-N against the trained detector."""
    sel = np.where((te["letter"] == LETTER_D) & (te["type"] == 1))[0]
    M, K = len(sel), cfg["evaluation"]["dstar_candidates"]
    h_c, d_c = arg.sample(M * K)
    valid = torch.tensor(valid_typing(h_c, d_c, cfg["typing"]["bounds_s"])).view(M, K)
    seq, _ = tnorm.transform(h_c, d_c, use_flight)
    seq = torch.tensor(seq)
    st = torch.tensor(sel)
    mot = te["mot"][st].repeat_interleave(K, 0)
    logt = te["logt"][st].repeat_interleave(K, 0)
    logit = model(seq, mot, logt).view(M, K).masked_fill(~valid, float("-inf"))
    best = logit.argmax(1)
    flat = torch.arange(M) * K + best
    typ = seq[flat]
    p = predict(model, {"typ": typ, "mot": te["mot"][st], "logt": te["logt"][st]})
    base = 500_000_000 + fold_index * 1_000_000
    return {"logit": p["logit"], "contrib_typing": p["contrib_typing"], "contrib_motion": p["contrib_motion"],
            "contrib_otp": p["contrib_otp"], "y": np.zeros(M), "letter": np.full(M, 6), "type": np.full(M, 1),
            "cluster": base + np.arange(M) // 50, "gate": np.ones(M, dtype=bool),
            "otp_s": te["otp_s"][sel], "dataset": np.full(M, -1), "subject": np.full(M, -1),
            "h": h_c.reshape(M, K, 10)[np.arange(M), best.numpy()], "d": d_c.reshape(M, K, 9)[np.arange(M), best.numpy()]}
