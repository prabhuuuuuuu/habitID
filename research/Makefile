# One-command reproduction from raw data to paper assets (CPU only; see README.md).
# Windows without make: run  powershell -ExecutionPolicy Bypass -File run_all.ps1
PY ?= python
export PYTHONPATH := src

all: test audit canonical folds baselines cv timing_probe train eval transfer onnx paper qa submission

qa:
	$(PY) scripts/06_verify_runs.py
	$(PY) scripts/11_typing_only_diagnosis.py
	$(PY) scripts/07b_cap_sensitivity.py
	$(PY) scripts/12_latex_checks.py

submission:
	$(PY) scripts/13_build_submission.py

test:
	$(PY) -m pytest -q

audit:
	$(PY) scripts/00_audit_data.py

canonical:
	$(PY) scripts/01_build_canonical.py

folds:
	$(PY) scripts/02_build_folds_bots.py

baselines:
	$(PY) scripts/03_baselines.py
	$(PY) scripts/03b_baselines_no_otp.py
	$(PY) scripts/03c_gbt_lofo.py

cv:
	$(PY) scripts/04_cv_reliability.py

timing_probe:
	$(PY) scripts/05b_timing_probe.py

train:
	$(PY) scripts/05_train_habitid.py --group main
	$(PY) scripts/05_train_habitid.py --group ablation
	$(PY) scripts/05_train_habitid.py --group lofo

eval:
	$(PY) scripts/07_eval_main.py

transfer:
	$(PY) scripts/08b_transfer_diagnostics.py
	$(PY) scripts/08_transfer_otp.py otp

onnx:
	$(PY) scripts/09_onnx_ondevice.py

paper:
	$(PY) scripts/10_make_paper_assets.py
