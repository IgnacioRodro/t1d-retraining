"""Environment report (second phase of script 00, run inside the project environment).

Reads what the setup phase recorded (package probes, GPU, actions) and writes
the tables, figures and summary of ``results/00_check_environment/``.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .compute import ComputeResources, detect_resources, lightning_trainer_kwargs
from .config import PROJECT_ROOT, derived_data_dir, load_paths
from .outputs import ScriptOutput
from .plotting import STATUS_COLORS, set_style

SCRIPT_ID = "00_check_environment"
SETUP_FILES = ["setup_steps.csv", "probe_before.csv", "probe_after.csv", "pip_log.txt",
               "nvidia_smi.txt", "setup_summary.json"]


def hardware_table(resources: ComputeResources, gpu: dict) -> pd.DataFrame:
    """Python, operating system, CPU, memory and accelerator information."""
    info = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count_logical": resources.cpu_logical,
        "cpu_count_physical": resources.cpu_physical,
        "ram_gb": resources.ram_gb,
        "nvidia_driver": gpu.get("driver") or "none",
        "driver_cuda_version": gpu.get("cuda_version") or "none",
        "torch_runtime_version": resources.torch_version or "not installed",
        "device": resources.device,
        "gpus_seen_by_torch": "; ".join(f"{g.name} ({g.memory_gb:.1f} GB)" for g in resources.gpus) or "none",
    }
    try:
        import torch

        info["torch_cuda_build"] = torch.version.cuda or "cpu-only build"
    except ImportError:
        pass
    return pd.DataFrame({"item": list(info), "value": [str(v) for v in info.values()]})


def compute_plan_table(resources: ComputeResources) -> pd.DataFrame:
    """How the project will use this machine (rule R5)."""
    trainer = lightning_trainer_kwargs(resources)
    plan = {
        "cores_available": resources.cpu_available,
        "cores_source": resources.source_of_cores,
        "reserve_cores": resources.reserve_cores,
        "parallel_workers": resources.n_workers,
        "device": resources.device,
        "n_gpus": resources.n_gpus,
        "lightning_accelerator": trainer["accelerator"],
        "lightning_devices": trainer["devices"],
    }
    return pd.DataFrame({"item": list(plan), "value": [str(v) for v in plan.values()]})


def dataset_folders_table(paths: dict[str, Path]) -> pd.DataFrame:
    """Existence, size and number of files of each configured dataset folder."""
    rows = []
    for key, folder in paths.items():
        exists = folder.exists()
        n_files, n_zip, size = 0, 0, 0
        if exists:
            for path in folder.rglob("*"):
                if path.is_file():
                    n_files += 1
                    size += path.stat().st_size
                    n_zip += path.suffix.lower() == ".zip"
        rows.append({"dataset": key, "path": str(folder), "exists": exists, "n_files": n_files,
                     "n_zip_files": n_zip, "size_gb": round(size / 1e9, 3)})
    return pd.DataFrame(rows, columns=["dataset", "path", "exists", "n_files", "n_zip_files", "size_gb"])


def storage_table(paths_file: Path) -> pd.DataFrame:
    """Free disk space where the project, the derived data and the environment live."""
    rows = []
    for role, folder in {"project": PROJECT_ROOT, "derived_data": derived_data_dir(paths_file),
                         "python_environment": Path(sys.prefix)}.items():
        probe = folder if folder.exists() else folder.parent
        free = shutil.disk_usage(probe).free / 1e9 if probe.exists() else None
        rows.append({"role": role, "path": str(folder), "exists": folder.exists(),
                     "free_disk_gb": round(free, 1) if free is not None else None})
    return pd.DataFrame(rows)


def plot_package_status(packages: pd.DataFrame) -> plt.Figure:
    """One bar per package, coloured by the result of the import test."""
    frame = packages.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(frame) + 1.2))
    ax.barh(frame["package"], [1] * len(frame), color=[STATUS_COLORS.get(s, "#adb5bd") for s in frame["status"]])
    for y, (_, row) in enumerate(frame.iterrows()):
        label = row["installed"] if isinstance(row["installed"], str) and row["installed"] else "not installed"
        ax.text(0.02, y, f"{label}  (required: {row['requirement']})", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.grid(False)
    present = [s for s in STATUS_COLORS if s in set(frame["status"])]
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS_COLORS[s]) for s in present]
    ax.legend(handles, present, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=len(present))
    ax.set_title("Package import test inside the project environment", loc="left", pad=24)
    return fig


def plot_dataset_sizes(folders: pd.DataFrame) -> plt.Figure:
    """Size on disk per dataset folder (missing folders in red)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [STATUS_COLORS["ok"] if e else STATUS_COLORS["missing"] for e in folders["exists"]]
    ax.bar(folders["dataset"], folders["size_gb"], color=colors)
    ax.set_ylabel("Size on disk (GB)")
    ax.set_title("Dataset folders configured in config/paths.yaml", loc="left")
    ax.tick_params(axis="x", rotation=30)
    return fig


def build_summary(setup: dict, packages: pd.DataFrame, resources: ComputeResources,
                  folders: pd.DataFrame, storage: pd.DataFrame, steps: pd.DataFrame) -> str:
    """Markdown digest with explicit warnings."""
    lines = ["# 00 - Environment setup and check", ""]
    info = setup["info"]
    lines.append(f"- Environment: {info['manager']} `{info['env_name']}` at `{info['prefix']}`.")
    lines.append(f"- Python {sys.version.split()[0]} on {platform.platform()} ({platform.node()}).")
    lines.append(f"- Compute plan (R5): {resources.describe()}.")
    before = sum(p["status"] == "ok" for p in setup["before"])
    after = int((packages["status"] == "ok").sum())
    lines.append(f"- Packages importable: {before}/{len(setup['before'])} before setup, {after}/{len(packages)} after.")
    installs = steps[steps["step"].str.startswith(("install", "create", "uninstall"))]
    if len(installs):
        lines.append("- Setup actions: " + "; ".join(f"{r.step} ({r.status}, {r.seconds:.0f} s)"
                                                     for r in installs.itertuples()) + ".")
    warnings = []
    problems = packages[packages["status"] != "ok"]
    for row in problems.itertuples():
        warnings.append(f"{row.package}: {row.status} (installed {row.installed or '-'}, required "
                        f"`{row.requirement}`) {row.error or ''}".strip())
    gpu = setup["gpu"]
    if gpu.get("available") and resources.device != "cuda":
        warnings.append("NVIDIA GPU present but torch cannot use it; rerun with --torch cu128 (or cu126) "
                        "or check the driver.")
    if not gpu.get("available"):
        warnings.append("No NVIDIA GPU visible. On a cluster, run this script on a GPU node to install the "
                        "CUDA build of torch.")
    if setup.get("pip_check") and "No broken requirements" not in setup["pip_check"]:
        warnings.append("pip check reports conflicts (see pip_log.txt): " + setup["pip_check"][:400])
    if len(folders):
        lines.append(f"- Dataset folders: {int(folders['exists'].sum())} of {len(folders)} found.")
        missing = folders.loc[~folders["exists"], "dataset"].tolist()
        if missing:
            warnings.append(f"Dataset folders not found: {', '.join(missing)} (check config/paths.yaml).")
    derived = storage[storage["role"] == "derived_data"].iloc[0]
    if derived["free_disk_gb"] is not None and derived["free_disk_gb"] < 100:
        warnings.append(f"Only {derived['free_disk_gb']} GB free for derived data; Loop alone needs tens of GB.")
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in warnings]
    return "\n".join(lines) + "\n"


def run_report(paths_file: Path) -> None:
    """Second phase of script 00."""
    set_style()
    resources = detect_resources()
    out = ScriptOutput(SCRIPT_ID, resources=resources, fresh=False)  # keeps the files of phase 1
    setup = json.loads((out.dir / "setup_summary.json").read_text(encoding="utf-8"))
    steps = pd.read_csv(out.dir / "setup_steps.csv")
    packages = pd.DataFrame(setup["after"])[["package", "requirement", "installed", "status", "import_ok",
                                             "version_ok", "module", "error"]]
    packages["installed"] = packages["installed"].fillna("")
    try:
        folders = dataset_folders_table(load_paths(paths_file))
    except FileNotFoundError as error:
        print(f"WARNING: {error}")
        folders = dataset_folders_table({})
    storage = storage_table(paths_file)

    out.csv(packages, "packages.csv")
    out.csv(hardware_table(resources, setup["gpu"]), "hardware.csv")
    out.csv(compute_plan_table(resources), "compute_plan.csv")
    out.csv(folders, "dataset_folders.csv")
    out.csv(storage, "storage.csv")
    out.figure(plot_package_status(packages), "environment_status.png")
    if len(folders):
        out.figure(plot_dataset_sizes(folders), "dataset_sizes.png")
    summary = build_summary(setup, packages, resources, folders, storage, steps)
    out.text(summary, "summary.md")
    for name in SETUP_FILES:
        out.register(name)
    out.manifest(settings={"paths_file": str(paths_file), "environment": setup["info"]})
    print(summary)
    print(f"Results written to {out.dir}")
