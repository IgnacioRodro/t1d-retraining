"""Delete the result caches of scripts 20, 30, 40, 42, 43, 50, 71-73, 75 and 93 (``cache`` folders in
<derived>/runs/<script>).

``tools/run_all.sh`` calls it before a full run (from script 02), so that every result of that run is computed by
the current release (D90), while a run resumed after a failure still reuses what the same run already computed.

Usage:  python tools/clear_caches.py [--paths config/paths.yaml]
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain.config import CONFIG_DIR, derived_data_dir  # noqa: E402

CACHED = ("20_stage1_ageing_runs", "30_stage2_policy_runs", "40_stage3_mechanism_runs", "42_stage3b_physiology",
          "43_stage3_context", "50_confirmation_runs", "71_low_glucose_loss", "72_change_recovery", "73_wearables",
          "75_precedent_is_lstm", "93_method_validation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    derived = derived_data_dir(parser.parse_args().paths)
    for script in CACHED:
        root = derived / "runs" / script
        for folder in sorted(root.rglob("cache")) if root.exists() else []:
            if folder.is_dir():
                shutil.rmtree(folder)
                print(f"removed {folder}")
    print("result caches cleared")


if __name__ == "__main__":
    main()
