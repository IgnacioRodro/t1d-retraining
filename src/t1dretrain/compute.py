"""Compute resources: CPU workers, GPUs and parallel execution (rule R5).

Every script detects its environment at start-up and:

- runs independent work in parallel on all available cores except a reserve
  (two by default, ``compute.reserve_cores`` in ``config/project.yaml``);
- sends to the GPU whatever can run there (CUDA first, then Apple MPS);
- records the resources it used in ``run_manifest.json``.

Available cores respect CPU affinity, cgroup limits and SLURM allocations, so
the same code behaves correctly on a laptop and on a shared compute server.
The number of workers can be forced with the environment variable
``T1D_N_WORKERS`` (useful for debugging with 1 worker).

The module imports only the standard library at load time, so the environment
setup (script 00, before any third-party package exists) can reuse
``available_cores`` and ``default_workers``.
"""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence

DEFAULT_RESERVE_CORES = 2


@dataclass
class GPUInfo:
    index: int
    name: str
    memory_gb: float


@dataclass
class ComputeResources:
    """What the current machine offers and how the project will use it."""

    cpu_logical: int
    cpu_available: int
    cpu_physical: int | None
    reserve_cores: int
    n_workers: int
    ram_gb: float | None
    torch_version: str | None
    device: str  # "cuda", "mps" or "cpu"
    gpus: list[GPUInfo] = field(default_factory=list)
    source_of_cores: str = "os"

    @property
    def n_gpus(self) -> int:
        return len(self.gpus)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["n_gpus"] = self.n_gpus
        return data

    def describe(self) -> str:
        """One line for logs and summaries."""
        gpu_text = ", ".join(f"{g.name} ({g.memory_gb:.0f} GB)" for g in self.gpus) or "none"
        return (
            f"{self.cpu_available} cores available ({self.cpu_logical} logical, source: {self.source_of_cores}); "
            f"{self.n_workers} parallel workers (reserve {self.reserve_cores}); "
            f"device: {self.device}; GPUs: {gpu_text}; RAM: {self.ram_gb} GB"
        )


def available_cores() -> tuple[int, str]:
    """Cores this process may use: SLURM allocation, else CPU affinity, else all logical cores."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return int(slurm), "slurm"
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0)), "affinity"
        except OSError:
            pass
    return os.cpu_count() or 1, "os"


def default_workers(reserve_cores: int = DEFAULT_RESERVE_CORES) -> int:
    """Workers for parallel work: available cores minus the reserve, or ``T1D_N_WORKERS``."""
    forced = os.environ.get("T1D_N_WORKERS")
    if forced and forced.isdigit():
        return max(1, int(forced))
    return max(1, available_cores()[0] - reserve_cores)


def detect_gpus() -> tuple[str, list[GPUInfo], str | None]:
    """Preferred torch device and the list of CUDA GPUs (empty without torch or CUDA)."""
    try:
        import torch
    except ImportError:
        return "cpu", [], None
    gpus: list[GPUInfo] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(GPUInfo(index, props.name, props.total_memory / 1e9))
        return "cuda", gpus, torch.__version__
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", gpus, torch.__version__
    return "cpu", gpus, torch.__version__


def detect_resources(reserve_cores: int | None = None) -> ComputeResources:
    """Inspect the machine and decide how many workers and which device to use."""
    from .config import load_project_config  # needs PyYAML, so imported here and not at module load

    settings = load_project_config().get("compute", {})
    reserve = settings.get("reserve_cores", DEFAULT_RESERVE_CORES) if reserve_cores is None else reserve_cores
    cores, source = available_cores()
    n_workers = max(1, cores - reserve)
    forced = os.environ.get("T1D_N_WORKERS")
    if forced and forced.isdigit():
        n_workers, source = max(1, int(forced)), f"{source}+T1D_N_WORKERS"
    try:
        import psutil

        ram = round(psutil.virtual_memory().total / 1e9, 1)
        physical = psutil.cpu_count(logical=False)
    except ImportError:
        ram, physical = None, None
    device, gpus, torch_version = detect_gpus()
    if not settings.get("prefer_gpu", True):
        device = "cpu"
    return ComputeResources(
        cpu_logical=os.cpu_count() or 1,
        cpu_available=cores,
        cpu_physical=physical,
        reserve_cores=reserve,
        n_workers=n_workers,
        ram_gb=ram,
        torch_version=torch_version,
        device=device,
        gpus=gpus,
        source_of_cores=source,
    )


def lightning_trainer_kwargs(resources: ComputeResources) -> dict[str, Any]:
    """Accelerator settings for PyTorch Lightning models (neuralforecast)."""
    if resources.device == "cuda":
        return {"accelerator": "gpu", "devices": resources.n_gpus or 1}
    if resources.device == "mps":
        return {"accelerator": "mps", "devices": 1}
    return {"accelerator": "cpu", "devices": 1}


def parallel_map(
    function: Callable[[Any], Any],
    items: Sequence[Any] | Iterable[Any],
    n_workers: int,
    label: str = "tasks",
    threads_per_worker: int = 1,
) -> list[Any]:
    """Apply ``function`` to every item on ``n_workers`` processes, keeping the input order.

    ``function`` must be defined at module level (Windows uses process spawning).
    Each worker limits BLAS/OpenMP threads to ``threads_per_worker`` to avoid
    oversubscribing the machine. With one worker, or a single item, the work
    runs in the current process, which keeps debugging simple.
    """
    items = list(items)
    if n_workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    from joblib import Parallel, delayed, parallel_config

    print(f"Running {len(items)} {label} on {n_workers} workers")
    try:
        with parallel_config(backend="loky", inner_max_num_threads=threads_per_worker):
            return Parallel(n_jobs=n_workers, verbose=5 if len(items) > 50 else 0)(
                delayed(function)(item) for item in items
            )
    finally:
        stop_workers()


def stop_workers() -> None:
    """Close the worker processes of ``parallel_map`` (joblib keeps them for reuse): they release their GPU memory
    and are not left behind when the script ends or is stopped (it. 25: idle workers of an earlier run survived it
    for hours, holding GPU memory)."""
    try:
        from joblib.externals.loky import reusable_executor

        executor = getattr(reusable_executor, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, kill_workers=True)
    except Exception:  # noqa: BLE001  (best effort: never fail a run here)
        pass


def machine_label() -> str:
    """Short host description for manifests."""
    return f"{platform.node()} | {platform.system()} {platform.release()} | {platform.machine()}"


# ----------------------------------------------------------------------------- GPU sharing between worker processes
#
# Many worker processes share few GPUs (38 workers and 2 GPUs on the server). Two rules keep them within the GPU
# memory (D73): every worker process uses one GPU only (``worker_device``: the GPU with the fewest workers of the run
# when the process first asks), so it holds a single CUDA context; and at most ``T1D_GPU_SLOTS`` processes per GPU
# (``compute.gpu_slots_per_gpu``, default 6) compute on it at the same time (``gpu_slot``, a file lock per slot; the
# lock is released if a process dies). A process that leaves its slot returns the cached GPU memory to the GPU.

_SLOT_STATE = {"depth": 0}
_WORKER = {}


def _lock_folder():
    import tempfile
    from pathlib import Path

    user = os.getuid() if hasattr(os, "getuid") else 0
    folder = Path(os.environ.get("T1D_GPU_LOCK_DIR", Path(tempfile.gettempdir()) / f"t1dretrain-gpu-{user}"))
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _least_used_gpu(count: int) -> int:
    """The GPU with the fewest worker processes of this run so far (a counter file per parent process)."""
    try:
        import fcntl
        import json
    except ImportError:
        return os.getpid() % count
    path = _lock_folder() / f"assign-{os.getppid()}.json"
    with open(path, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        text = handle.read()
        counts = json.loads(text) if text.strip() else {}
        used = [int(counts.get(str(k), 0)) for k in range(count)]
        index = used.index(min(used))
        counts[str(index)] = used[index] + 1
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(counts))
        fcntl.flock(handle, fcntl.LOCK_UN)
    return index


def share_gpus(config: dict | None = None) -> int:
    """Set the number of processes that may compute on one GPU at the same time (``compute.gpu_slots_per_gpu``,
    default 6; the environment variable ``T1D_GPU_SLOTS`` wins) and start a fresh GPU assignment for the workers of
    this process. Called by every script that runs networks before its workers start."""
    value = (config or {}).get("compute", {}).get("gpu_slots_per_gpu", 6)
    os.environ.setdefault("T1D_GPU_SLOTS", str(int(value)))
    try:
        (_lock_folder() / f"assign-{os.getpid()}.json").unlink(missing_ok=True)
    except OSError:
        pass
    return int(os.environ["T1D_GPU_SLOTS"])


def worker_device(choice: str) -> str:
    """``cuda`` -> the one GPU of this worker process, chosen once as the GPU with the fewest workers of the run
    (D73); any other choice is returned as is."""
    if choice != "cuda":
        return choice
    if "device" not in _WORKER:
        try:
            import torch

            count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:  # noqa: BLE001
            count = 0
        _WORKER["device"] = f"cuda:{_least_used_gpu(count)}" if count else "cpu"
    return _WORKER["device"]


class gpu_slot:
    """Context manager: wait for a free slot of the GPU of ``device`` (no-op on the CPU, on systems without
    ``fcntl`` or when ``T1D_GPU_SLOTS`` is 0). Re-entrant within a process."""

    def __init__(self, device: str) -> None:
        self.device = str(device)
        self.handle = None

    def __enter__(self) -> "gpu_slot":
        _SLOT_STATE["depth"] += 1
        if _SLOT_STATE["depth"] > 1 or not self.device.startswith("cuda"):
            return self
        try:
            import fcntl
        except ImportError:  # Windows: no slots
            return self
        import time

        slots = int(os.environ.get("T1D_GPU_SLOTS", "6"))
        if slots <= 0:
            return self
        index = int(self.device.split(":")[1]) if ":" in self.device else 0
        folder = _lock_folder()
        while self.handle is None:
            for slot in range(slots):
                handle = open(folder / f"gpu{index}-slot{slot}.lock", "a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.handle = handle
                    break
                except OSError:
                    handle.close()
            if self.handle is None:
                time.sleep(0.2)
        return self

    def __exit__(self, *exc) -> None:
        _SLOT_STATE["depth"] -= 1
        if self.handle is not None:
            try:
                import torch

                torch.cuda.empty_cache()  # give the cached memory back before another process takes the slot
            except Exception:  # noqa: BLE001
                pass
            import fcntl

            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
