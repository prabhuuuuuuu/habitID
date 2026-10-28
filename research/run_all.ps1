# One-command reproduction (Windows PowerShell). Equivalent to the Makefile targets.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = "src"
$steps = @(
  @("-m", "pytest", "-q"),
  @("scripts/00_audit_data.py"),
  @("scripts/01_build_canonical.py"),
  @("scripts/02_build_folds_bots.py"),
  @("scripts/03_baselines.py"),
  @("scripts/04_cv_reliability.py"),
  @("scripts/05b_timing_probe.py"),
  @("scripts/05_train_habitid.py", "--group", "main"),
  @("scripts/05_train_habitid.py", "--group", "ablation"),
  @("scripts/05_train_habitid.py", "--group", "lofo"),
  @("scripts/07_eval_main.py"),
  @("scripts/08_transfer_otp.py"),
  @("scripts/09_onnx_ondevice.py"),
  @("scripts/10_make_paper_assets.py")
)
foreach ($s in $steps) {
  Write-Host ("== python " + ($s -join " ")) -ForegroundColor Cyan
  python @s
  if ($LASTEXITCODE -ne 0) { throw "step failed: $($s -join ' ')" }
}
