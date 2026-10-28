# Parallel training plan. Nine workers, two torch threads each (18 threads on 18 logical CPUs).
# Each worker runs its commands in sequence. Finished runs are skipped (resumable).
# Epoch cap and patience come from configs/default.yaml (training section).
# Logs: research/results/timing/worker_<name>.log
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = "src"
New-Item -ItemType Directory -Force -Path results\timing | Out-Null
$py = "scripts\05_train_habitid.py"
# Each worker: list of argument lists, run in order.
$workers = [ordered]@{
  "w1_full01"       = @(@($py, "--variants", "full", "--seeds", "0", "1"))
  "w2_full2_typOtp" = @(@($py, "--variants", "full", "--seeds", "2"), @($py, "--variants", "typing_only", "otp_only"))
  "w3_noprior01"    = @(@($py, "--variants", "noprior", "--seeds", "0", "1"))
  "w4_noprior2_noadv" = @(@($py, "--variants", "noprior", "--seeds", "2"), @($py, "--variants", "noadv"))
  "w5_motion_nomo"  = @(@($py, "--variants", "motion_only"), @($py, "--variants", "no_motion"))
  "w6_lofoA_notyp"  = @(@($py, "--variants", "lofo_full_A", "lofo_noadv_A"), @($py, "--variants", "no_typing"))
  "w7_lofoB_C"      = @(@($py, "--variants", "lofo_full_B", "lofo_noadv_B", "lofo_noadv_C"))
  "w8_lofoC_D_lam10" = @(@($py, "--variants", "lofo_full_C", "lofo_noadv_D"), @($py, "--variants", "lam10"))
  "w9_lofoD_nootp"  = @(@($py, "--variants", "lofo_full_D"), @($py, "--variants", "no_otp"))
}
$procs = @()
foreach ($name in $workers.Keys) {
  $cmds = $workers[$name]
  $chain = ($cmds | ForEach-Object { "python " + (($_ + @("--threads", "2")) -join " ") }) -join " && "
  $log = "results\timing\worker_$name.log"
  $procs += Start-Process -NoNewWindow -PassThru -FilePath cmd.exe -ArgumentList "/c", "($chain) > $log 2>&1"
  Write-Host "started $name -> $log"
}
$procs | ForEach-Object { $_.WaitForExit() }
Write-Host "all workers finished"
