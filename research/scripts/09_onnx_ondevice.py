"""Phase 7: ONNX export of habitID-Net (full variant, seed 0, each fold), onnxruntime CPU latency,
dynamic int8 quantization (size and accuracy change). No browser page, no mobile-device numbers.

Latency: single-session inference, 100 warm-up runs, N = 1000 timed runs, onnxruntime with intra_op_num_threads=1.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402

from habitid.calibrate.calibration import p_human  # noqa: E402
from habitid.eval.metrics import detection_summary, ece  # noqa: E402
from habitid.features.features import MotionNorm, TypingNorm  # noqa: E402
from habitid.models.habitid import HabitIDNet  # noqa: E402
from habitid.train.train import tensors_from_split  # noqa: E402
from habitid.utils.config import PROC, RESULTS, load_cfg, load_otp  # noqa: E402
from habitid.utils.runtime import phase_timer, write_json  # noqa: E402

ONNX_DIR = RESULTS / "ondevice"


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, typing, motion, log_otp):
        return self.m(typing, motion, log_otp)


def load_ckpt(k: int, otp_cfg: dict, cfg: dict):
    ck = torch.load(RESULTS / "checkpoints" / "full" / f"fold{k}_seed0.pt", weights_only=False)
    m = HabitIDNet(ck["active"], {a: v for a, v in zip(ck["active"], [25 / 60, 20 / 60, 15 / 60])}, otp_cfg,
                   cfg["model"]["lstm_hidden"], cfg["model"]["lstm_layers"], cfg["model"]["cnn_channels"])
    m.load_state_dict(ck["state"])
    m.eval()
    tn = TypingNorm(ck["tnorm"]["scale"])
    tn.mu = {kk: np.array(v) for kk, v in ck["tnorm"]["mu"].items()}
    tn.sd = {kk: np.array(v) for kk, v in ck["tnorm"]["sd"].items()}
    mn = MotionNorm()
    mn.mean = np.array(ck["mnorm"]["mean"], dtype=np.float32)
    mn.std = np.array(ck["mnorm"]["std"], dtype=np.float32)
    return m, tn, mn, ck["T"], ck["use_flight"]


def export(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    typ = torch.zeros(1, 10, 4); mot = torch.zeros(1, 150, 6); lo = torch.zeros(1)
    torch.onnx.export(Wrap(model), (typ, mot, lo), str(path), input_names=["typing", "motion", "log_otp"],
                      output_names=["logit"], dynamic_axes={"typing": {0: "b"}, "motion": {0: "b"},
                                                            "log_otp": {0: "b"}, "logit": {0: "b"}},
                      opset_version=17, dynamo=False)


def ort_session(path: Path, threads: int = 1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run_ort(sess, typ, mot, lo, bs=2048):
    out = []
    for i in range(0, len(typ), bs):
        out.append(sess.run(["logit"], {"typing": typ[i:i + bs], "motion": mot[i:i + bs],
                                        "log_otp": lo[i:i + bs]})[0])
    return np.concatenate(out)


def latency(sess, typ, mot, lo, n=1000, warm=100):
    idx = np.arange(warm + n) % len(typ)
    for i in idx[:warm]:
        sess.run(["logit"], {"typing": typ[i:i + 1], "motion": mot[i:i + 1], "log_otp": lo[i:i + 1]})
    times = []
    for i in idx[warm:]:
        t0 = time.perf_counter()
        sess.run(["logit"], {"typing": typ[i:i + 1], "motion": mot[i:i + 1], "log_otp": lo[i:i + 1]})
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
            "mean_ms": float(t.mean()), "n_timed": int(n), "warmup": int(warm)}


def metrics_from_logit(logit, label, T, cfg):
    bot = 1 - label
    m = detection_summary(bot, -logit, cfg["evaluation"]["base_rates"], cfg["evaluation"]["fpr_target"])
    m["ECE"] = ece(p_human(logit, T), label, cfg["evaluation"]["ece_bins"])
    return m


def main():
    cfg = load_cfg()
    otp_cfg = load_otp()
    results = {"per_fold": [], "latency": {}, "note": "single-session CPU inference on this machine; not a mobile measurement"}
    with phase_timer("phase7_onnx"):
        for k in range(cfg["project"]["n_folds"]):
            m, tn, mn, T, use_flight = load_ckpt(k, otp_cfg, cfg)
            fold = dict(np.load(PROC / "folds" / f"fold{k}.npz"))
            fold["fold_index"] = np.array(k)
            tt = tensors_from_split(fold, "test", tn, mn, use_flight, keep=fold["test_gate"].astype(bool))
            typ, mot, lo = tt["typ"].numpy(), tt["mot"].numpy(), tt["logt"].numpy()
            label = tt["y"].numpy().astype(int)
            with torch.no_grad():
                logit_t = m(tt["typ"], tt["mot"], tt["logt"]).numpy()
            fp = ONNX_DIR / f"habitid_full_fold{k}_fp32.onnx"
            i8 = ONNX_DIR / f"habitid_full_fold{k}_int8.onnx"
            export(m, fp)
            try:
                quantize_dynamic(str(fp), str(i8), weight_type=QuantType.QInt8,
                                 op_types_to_quantize=["MatMul", "Gemm", "LSTM"])
                q_ops = ["MatMul", "Gemm", "LSTM"]
            except Exception as e:  # fall back to dense layers only
                quantize_dynamic(str(fp), str(i8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
                q_ops = ["MatMul", "Gemm"] + [f"fallback({type(e).__name__})"]
            s32 = ort_session(fp); s8 = ort_session(i8)
            l32 = run_ort(s32, typ, mot, lo).reshape(-1)
            l8 = run_ort(s8, typ, mot, lo).reshape(-1)
            rec = {
                "fold": k, "n_test_gate_pass": int(len(label)),
                "max_abs_logit_diff_torch_vs_fp32": float(np.abs(logit_t - l32).max()),
                "fp32": metrics_from_logit(l32, label, T, cfg),
                "int8": metrics_from_logit(l8, label, T, cfg),
                "int8_logit_mad_vs_fp32": float(np.abs(l8 - l32).mean()),
                "decision_agreement_p0.5": float(((p_human(l8, T) >= 0.5) == (p_human(l32, T) >= 0.5)).mean()),
                "quantized_ops": q_ops,
                "size_bytes": {"fp32": os.path.getsize(fp), "int8": os.path.getsize(i8)},
            }
            results["per_fold"].append(rec)
            print(k, {x: round(rec["fp32"][x], 4) for x in ("AUC", "TPR@1%FPR", "ECE")},
                  {x: round(rec["int8"][x], 4) for x in ("AUC", "TPR@1%FPR", "ECE")}, flush=True)
            if k == 0:
                results["latency"]["fp32"] = latency(s32, typ, mot, lo)
                results["latency"]["int8"] = latency(s8, typ, mot, lo)
                results["latency"]["torch_cpu_reference"] = None
    pf = results["per_fold"]
    results["summary"] = {
        "size_kb_fp32": float(np.mean([r["size_bytes"]["fp32"] for r in pf]) / 1024),
        "size_kb_int8": float(np.mean([r["size_bytes"]["int8"] for r in pf]) / 1024),
        "AUC_change_int8_minus_fp32": float(np.mean([r["int8"]["AUC"] - r["fp32"]["AUC"] for r in pf])),
        "TPR_change_int8_minus_fp32": float(np.mean([r["int8"]["TPR@1%FPR"] - r["fp32"]["TPR@1%FPR"] for r in pf])),
        "ECE_change_int8_minus_fp32": float(np.mean([r["int8"]["ECE"] - r["fp32"]["ECE"] for r in pf])),
        "decision_agreement_mean": float(np.mean([r["decision_agreement_p0.5"] for r in pf])),
    }
    write_json(RESULTS / "ondevice" / "ondevice.json", results)
    print(results["summary"], results["latency"])


if __name__ == "__main__":
    main()
