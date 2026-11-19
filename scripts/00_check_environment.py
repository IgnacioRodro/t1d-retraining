"""00 - Create or repair the project environment, then check the machine.

Run it with any Python 3.8+ that can see conda (or mamba / micromamba); no
third-party package is needed to start. It works in two phases:

Phase 1 (setup, current Python, standard library only)
    1. Creates the conda environment of environment/environment.json
       (default: "t1d", Python 3.12, conda-forge) if it does not exist.
    2. Tries to import every package of environment/requirements.txt inside
       that environment, one probe per package, in parallel (rule R5).
    3. Installs what is missing, broken or outside its version range. Torch
       is installed first, as a CUDA build when nvidia-smi shows a GPU.
    4. Installs the project package (pip install -e .), runs pip check and
       records the environment in config/environment.local.json, which all
       other scripts use to run inside it automatically.

Phase 2 (report, re-launched inside the environment)
    Writes the package, hardware, compute-plan, dataset and storage reports.

Outputs (results/00_check_environment/):
    setup_steps.csv        every action taken (create, install, ...) with status and duration
    probe_before.csv       import test of every package before the setup
    probe_after.csv        import test after the setup
    pip_log.txt            output of conda and pip
    nvidia_smi.txt         raw nvidia-smi output (if any)
    packages.csv           final status of every package (ok, missing, wrong_version, import_error)
    hardware.csv           Python, OS, CPU, RAM, driver and GPU information
    compute_plan.csv       workers, device and GPUs the project will use on this machine
    dataset_folders.csv    whether each configured dataset folder exists, its size and file count
    storage.csv            free disk space for project, derived data and environment
    environment_status.png package status at a glance
    dataset_sizes.png      size on disk of each dataset folder
    summary.md             short digest with warnings
    run_manifest.json      run metadata

Usage:
    python scripts/00_check_environment.py                   # create/repair "t1d" and report
    python scripts/00_check_environment.py --recreate        # delete and rebuild the environment
    python scripts/00_check_environment.py --no-install      # only report what is missing
    python scripts/00_check_environment.py --torch cu128     # force a torch build (auto, cu130, cu128, cu126, cpu, pypi)
    python scripts/00_check_environment.py --env-prefix /scratch/me/envs/t1d   # environment at a path
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t1dretrain import envsetup  # noqa: E402  (standard library only)

RESULTS = ROOT / "results" / "00_check_environment"


def write_csv(rows: list[dict], path: Path) -> None:
    """Write a list of dictionaries as CSV with the standard library."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row}, key=lambda k: (k != "package", k != "step", k))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def phase_setup(args: argparse.Namespace) -> dict:
    """Create or repair the environment and record what was done."""
    settings = envsetup.load_env_settings()
    log = envsetup.SetupLog()
    env_name = args.env_name or settings["env_name"]
    manager = "venv" if args.venv else settings.get("manager", "conda")
    result = envsetup.setup_environment(
        manager=manager,
        env_name=env_name,
        env_prefix=Path(args.env_prefix).resolve() if args.env_prefix else None,
        recreate=args.recreate,
        install=not args.no_install,
        torch_choice=args.torch,
        requirements_file=Path(args.requirements).resolve() if args.requirements else None,
        log=log,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(log.steps, RESULTS / "setup_steps.csv")
    write_csv(result["before"], RESULTS / "probe_before.csv")
    write_csv(result["after"], RESULTS / "probe_after.csv")
    (RESULTS / "pip_log.txt").write_text("\n".join(log.pip_output), encoding="utf-8")
    (RESULTS / "nvidia_smi.txt").write_text(result["gpu"].get("raw", "") or "nvidia-smi not available\n",
                                            encoding="utf-8")
    (RESULTS / "setup_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def phase_report(args: argparse.Namespace) -> None:
    """Run inside the environment: full report with pandas and matplotlib."""
    from t1dretrain.config import CONFIG_DIR
    from t1dretrain.envreport import run_report

    run_report(Path(args.paths) if args.paths else CONFIG_DIR / "paths.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-name", help="conda environment name (default: environment/environment.json)")
    parser.add_argument("--env-prefix", help="create the environment at this path instead of by name")
    parser.add_argument("--recreate", action="store_true", help="delete and rebuild the environment")
    parser.add_argument("--no-install", action="store_true", help="only probe and report, install nothing")
    parser.add_argument("--torch", default="auto", choices=["auto", "cu130", "cu128", "cu126", "cpu", "pypi"],
                        help="torch build to install (auto: CUDA build matching the driver if a GPU is visible)")
    parser.add_argument("--requirements", help="alternative requirements file (default: environment/requirements.txt)")
    parser.add_argument("--venv", action="store_true", help="use a venv from the current Python instead of conda")
    parser.add_argument("--paths", help="dataset paths YAML (default: config/paths.yaml)")
    parser.add_argument("--report-only", action="store_true", help=argparse.SUPPRESS)  # phase 2, internal
    args = parser.parse_args()

    if args.report_only:
        phase_report(args)
        return

    from t1dretrain.codecheck import verify_code  # standard library only

    verify_code(strict=False)  # setup never stops; scripts 01+ stop if files are out of date

    try:
        result = phase_setup(args)
    except RuntimeError as error:
        sys.exit(f"Environment setup failed: {error}")

    python = result["info"]["python"]
    command = [python, str(Path(__file__).resolve()), "--report-only"]
    if args.paths:
        command += ["--paths", args.paths]
    print(f"\nPhase 2: report inside the environment ({python})")
    env = dict(os.environ, T1D_ENV_ACTIVE="1")
    report = subprocess.run(command, env=env)
    if report.returncode != 0:
        sys.exit("The report phase failed inside the environment; see the messages above and "
                 "results/00_check_environment/pip_log.txt.")


if __name__ == "__main__":
    main()
