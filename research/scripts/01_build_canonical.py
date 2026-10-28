"""Phase 1: canonical typing sessions (CMU + KeyRecs-fixed) and MotionSense sit/std windows."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from habitid.data.canonical import build_motion, build_typing  # noqa: E402
from habitid.utils.config import load_cfg  # noqa: E402
from habitid.utils.runtime import phase_timer  # noqa: E402

if __name__ == "__main__":
    cfg = load_cfg()
    with phase_timer("phase1_canonical"):
        tc = build_typing(cfg)
        mc = build_motion(cfg)
    print("typing:", {k: v for k, v in tc.items() if k in ("total_kept", "CMU", "KeyRecs-fixed")})
    print("motion windows:", mc["windows"], "per activity:", mc["windows_per_activity"])
