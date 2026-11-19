"""Creation and repair of the project environment (used by script 00).

This module uses the standard library only: it runs with whatever Python is
available on the machine, before the project environment exists. It:

1. finds conda (or mamba / micromamba);
2. creates the project environment if it does not exist (``environment/environment.json``);
3. tries to import every package of ``environment/requirements.txt`` inside that
   environment, one probe per package, in parallel (rule R5);
4. installs whatever is missing, broken or outside its version range, with the
   CUDA build of torch when an NVIDIA GPU is present;
5. installs the project package (``pip install -e .``);
6. records where the environment lives in ``config/environment.local.json``,
   which every other script reads to run inside it (see ``envguard``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .compute import default_workers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_SETTINGS_FILE = PROJECT_ROOT / "environment" / "environment.json"
LOCAL_ENV_FILE = PROJECT_ROOT / "config" / "environment.local.json"

PROBE_CODE = r"""
import importlib, importlib.metadata as md, json, sys
dist, module = sys.argv[1], sys.argv[2]
out = {"installed": None, "import_ok": False, "error": ""}
try:
    out["installed"] = md.version(dist)
except md.PackageNotFoundError:
    pass
try:
    importlib.import_module(module)
    out["import_ok"] = True
except Exception as error:
    out["error"] = (type(error).__name__ + ": " + str(error))[:300]
if dist == "torch" and out["import_ok"]:
    import torch
    out["torch_cuda"] = bool(torch.cuda.is_available())
    out["torch_cuda_build"] = torch.version.cuda or ""
print("@@PROBE@@" + json.dumps(out))
"""


# ----------------------------------------------------------------------------- settings and requirements


def load_env_settings() -> dict:
    """Versioned settings of the project environment."""
    return json.loads(ENV_SETTINGS_FILE.read_text(encoding="utf-8"))


def read_local_env() -> dict | None:
    """Machine-specific record of where the environment lives (written by script 00)."""
    if LOCAL_ENV_FILE.exists():
        return json.loads(LOCAL_ENV_FILE.read_text(encoding="utf-8"))
    return None


def write_local_env(info: dict) -> None:
    LOCAL_ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_ENV_FILE.write_text(json.dumps(info, indent=2), encoding="utf-8")


@dataclass
class Requirement:
    name: str           # distribution name as written
    spec: str           # full requirement line, e.g. "pandas==2.3.3"
    constraints: list   # [(operator, version), ...]
    module: str         # import name

    @property
    def key(self) -> str:
        return by_key_name(self.name)


def by_key_name(name: str) -> str:
    """Normalised distribution name used as dictionary key."""
    return name.lower().replace("_", "-")


def parse_requirements(path: Path, import_names: dict[str, str]) -> list[Requirement]:
    """Requirement lines (comments and blank lines ignored)."""
    requirements = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*(.*)$", line)
        if not match:
            continue
        name, _, rest = match.groups()
        constraints = re.findall(r"(==|>=|<=|!=|~=|>|<)\s*([0-9][0-9A-Za-z.\-+*]*)", rest)
        key = by_key_name(name)
        module = import_names.get(key, key.replace("-", "_"))
        requirements.append(Requirement(name=name, spec=line, constraints=constraints, module=module))
    return requirements


def _version_tuple(version: str) -> tuple:
    version = version.split("+", 1)[0]
    parts = []
    for piece in re.split(r"[.\-]", version):
        number = re.match(r"\d+", piece)
        parts.append(int(number.group()) if number else 0)
    return tuple(parts)


def satisfies(version: str | None, constraints: list) -> bool:
    """True when ``version`` meets every (operator, version) constraint."""
    if version is None:
        return False
    current = _version_tuple(version)
    for operator, target in constraints:
        wanted = _version_tuple(target.replace("*", "0"))
        size = max(len(current), len(wanted))
        a = current + (0,) * (size - len(current))
        b = wanted + (0,) * (size - len(wanted))
        if operator == "==" and "*" in target:
            prefix = len(_version_tuple(target.split("*")[0].rstrip(".")))
            ok = a[:prefix] == b[:prefix]
        else:
            ok = {
                "==": a == b, "!=": a != b, ">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b,
                "~=": a >= b and a[: len(wanted) - 1] == b[: len(wanted) - 1],
            }[operator]
        if not ok:
            return False
    return True


# ----------------------------------------------------------------------------- machine inspection


def find_conda() -> str | None:
    """Path of conda, mamba or micromamba (CONDA_EXE first)."""
    candidates = [os.environ.get("CONDA_EXE"), os.environ.get("MAMBA_EXE")]
    candidates += [shutil.which(name) for name in ("conda", "mamba", "micromamba")]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def nvidia_info() -> dict:
    """Driver, CUDA version supported by the driver and GPU list from nvidia-smi."""
    info = {"available": False, "driver": "", "cuda_version": "", "gpus": [], "raw": ""}
    smi = shutil.which("nvidia-smi")
    if not smi:
        return info
    try:
        header = subprocess.run([smi], capture_output=True, text=True, timeout=60)
        query = subprocess.run(
            [smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return info
    info["raw"] = header.stdout
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", header.stdout)
    info["cuda_version"] = cuda.group(1) if cuda else ""
    for line in query.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            info["gpus"].append({"name": parts[0], "memory": parts[1]})
            info["driver"] = parts[2]
    info["available"] = bool(info["gpus"])
    return info


def torch_index_candidates(gpu: dict, choice: str, indexes: dict[str, str]) -> list[tuple[str, str | None]]:
    """Ordered (label, index_url) options to install torch; None means the default PyPI index."""
    if choice == "pypi":
        return [("pypi", None)]
    if choice != "auto":
        return [(choice, indexes[choice])]
    if not gpu["available"]:
        return [("pypi", None)] if sys.platform == "darwin" else [("cpu", indexes["cpu"]), ("pypi", None)]
    driver_cuda = _version_tuple(gpu["cuda_version"] or "0")
    options = [(label, url) for label, url in indexes.items() if label != "cpu"]
    supported = [(label, url) for label, url in options
                 if _version_tuple(f"{label[2:4]}.{label[4:]}") <= driver_cuda]
    supported.sort(key=lambda item: _version_tuple(f"{item[0][2:4]}.{item[0][4:]}"), reverse=True)
    return supported + [("pypi", None)]


# ----------------------------------------------------------------------------- environment management


@dataclass
class SetupLog:
    """Everything script 00 did, for the results folder."""

    steps: list[dict] = field(default_factory=list)
    pip_output: list[str] = field(default_factory=list)

    def add(self, step: str, detail: str, status: str, seconds: float = 0.0) -> None:
        print(f"[setup] {step}: {status} - {detail}")
        self.steps.append({"step": step, "detail": detail, "status": status, "seconds": round(seconds, 1)})


def run(command: list[str], log: SetupLog, step: str, timeout: int = 7200) -> subprocess.CompletedProcess:
    """Run a command, keep its output in the log."""
    start = time.time()
    print(f"[setup] $ {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    log.pip_output.append(f"$ {' '.join(command)}\n{result.stdout[-20000:]}\n{result.stderr[-20000:]}\n")
    log.add(step, " ".join(command)[:300], "ok" if result.returncode == 0 else f"failed ({result.returncode})",
            time.time() - start)
    return result


def conda_env_prefix(conda: str, name: str) -> Path | None:
    """Prefix of a named conda environment, if it exists."""
    result = subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return None
    for prefix in json.loads(result.stdout).get("envs", []):
        if Path(prefix).name == name:
            return Path(prefix)
    return None


def env_python(prefix: Path) -> Path:
    """Python executable inside an environment prefix."""
    return prefix / "python.exe" if os.name == "nt" else prefix / "bin" / "python"


def ensure_conda_env(settings: dict, log: SetupLog, env_name: str, env_prefix: Path | None, recreate: bool) -> Path:
    """Create (or recreate) the conda environment and return its prefix."""
    conda = find_conda()
    if conda is None:
        raise RuntimeError("conda, mamba or micromamba not found. Load the conda module of the server "
                           "(e.g. 'module load anaconda') or install Miniforge, then run this script again.")
    log.add("conda", conda, "found")
    target = env_prefix or conda_env_prefix(conda, env_name)
    if target is not None and recreate and Path(target).exists():
        flag = ["-p", str(target)] if env_prefix else ["-n", env_name]
        run([conda, "env", "remove", "-y", *flag], log, "remove_env")
        target = None if not env_prefix else env_prefix
    if target is not None and env_python(Path(target)).exists():
        log.add("environment", str(target), "exists")
        return Path(target)
    location = ["-p", str(env_prefix)] if env_prefix else ["-n", env_name]
    result = run([conda, "create", "-y", *location, f"python={settings['python']}", "pip",
                  "-c", settings.get("conda_channel", "conda-forge"), "--override-channels"], log, "create_env")
    if result.returncode != 0:
        raise RuntimeError(f"conda create failed:\n{result.stderr[-3000:]}")
    prefix = env_prefix or conda_env_prefix(conda, env_name)
    if prefix is None:
        raise RuntimeError("The environment was created but conda does not list it.")
    return Path(prefix)


def ensure_venv(log: SetupLog, venv_dir: Path, recreate: bool) -> Path:
    """Fallback without conda: a venv built from the current Python."""
    if recreate and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if not env_python(venv_dir).exists():
        run([sys.executable, "-m", "venv", str(venv_dir)], log, "create_venv")
    else:
        log.add("environment", str(venv_dir), "exists")
    return venv_dir


def _probe_one(args: tuple[str, Requirement]) -> dict:
    python, requirement = args
    try:
        result = subprocess.run([python, "-c", PROBE_CODE, requirement.name, requirement.module],
                                capture_output=True, text=True, timeout=900)
        line = next((l for l in result.stdout.splitlines() if l.startswith("@@PROBE@@")), None)
        data = json.loads(line[len("@@PROBE@@"):]) if line else {"installed": None, "import_ok": False,
                                                                 "error": result.stderr.strip()[-300:]}
    except subprocess.TimeoutExpired:
        data = {"installed": None, "import_ok": False, "error": "import timed out"}
    data.update(package=requirement.name, module=requirement.module, requirement=requirement.spec)
    data["version_ok"] = satisfies(data.get("installed"), requirement.constraints)
    data["status"] = (
        "ok" if data["import_ok"] and data["version_ok"]
        else "missing" if data.get("installed") is None
        else "wrong_version" if not data["version_ok"]
        else "import_error"
    )
    return data


def probe_packages(python: Path, requirements: list[Requirement], n_workers: int) -> list[dict]:
    """Try to import every package inside the environment, in parallel (one process per package)."""
    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
        return list(pool.map(_probe_one, [(str(python), r) for r in requirements]))


def install_missing(python: Path, requirements: list[Requirement], probes: list[dict], gpu: dict,
                    torch_choice: str, indexes: dict[str, str], log: SetupLog) -> None:
    """Install torch first (CUDA build when possible), then every other package that is not ok."""
    pip = [str(python), "-m", "pip", "install", "--upgrade-strategy", "only-if-needed"]
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], log, "upgrade_pip")
    by_key = {r.key: r for r in requirements}
    probe_by_key = {by_key_name(p["package"]): p for p in probes}

    torch_req, torch_probe = by_key.get("torch"), probe_by_key.get("torch")
    if torch_req is not None and torch_probe is not None:
        cpu_only_with_gpu = gpu["available"] and torch_probe["status"] == "ok" and not torch_probe.get("torch_cuda")
        if torch_probe["status"] != "ok" or (cpu_only_with_gpu and torch_choice != "pypi"):
            if torch_probe.get("installed"):
                run([str(python), "-m", "pip", "uninstall", "-y", "torch"], log, "uninstall_torch")
            for label, url in torch_index_candidates(gpu, torch_choice, indexes):
                command = pip + [torch_req.spec] + (["--index-url", url] if url else [])
                if run(command, log, f"install_torch_{label}").returncode == 0:
                    break

    pending = [by_key[key].spec for key, p in probe_by_key.items() if p["status"] != "ok" and key != "torch"]
    if pending:
        result = run(pip + pending, log, "install_packages")
        if result.returncode != 0:  # fall back to one package at a time to isolate the failure
            for spec in pending:
                run(pip + [spec], log, f"install_{re.split(r'[<>=!~ ]', spec)[0]}")
    run([str(python), "-m", "pip", "install", "--no-deps", "-e", str(PROJECT_ROOT)], log, "install_project")


def pip_check(python: Path, log: SetupLog) -> str:
    """Dependency conflicts reported by pip."""
    result = run([str(python), "-m", "pip", "check"], log, "pip_check", timeout=600)
    return (result.stdout + result.stderr).strip()


def setup_environment(manager: str, env_name: str, env_prefix: Path | None, recreate: bool, install: bool,
                      torch_choice: str, requirements_file: Path | None, log: SetupLog) -> dict:
    """Create or repair the environment; return a summary with probes before and after."""
    settings = load_env_settings()
    import_names = {k.lower(): v for k, v in settings.get("import_names", {}).items()}
    requirements = parse_requirements(requirements_file or PROJECT_ROOT / settings["requirements"], import_names)
    n_workers = default_workers()
    gpu = nvidia_info()
    log.add("nvidia_smi", f"GPUs: {[g['name'] for g in gpu['gpus']]}; driver CUDA {gpu['cuda_version'] or '-'}",
            "found" if gpu["available"] else "no NVIDIA GPU")

    if manager == "conda":
        prefix = ensure_conda_env(settings, log, env_name, env_prefix, recreate)
    else:
        prefix = ensure_venv(log, env_prefix or (PROJECT_ROOT / ".venv-t1d"), recreate)
    python = env_python(prefix)

    start = time.time()
    before = probe_packages(python, requirements, n_workers)
    log.add("probe_before", f"{sum(p['status'] == 'ok' for p in before)}/{len(before)} packages ok",
            "done", time.time() - start)
    needs_work = any(p["status"] != "ok" for p in before) or not _project_installed(python)
    if gpu["available"] and any(p["package"].lower() == "torch" and p["status"] == "ok" and not p.get("torch_cuda")
                                for p in before):
        needs_work = True  # CPU-only torch on a GPU machine
    if install and needs_work:
        install_missing(python, requirements, before, gpu, torch_choice, settings["torch_cuda_indexes"], log)
    elif not install:
        log.add("install", "skipped (--no-install)", "skipped")
    start = time.time()
    after = probe_packages(python, requirements, n_workers)
    log.add("probe_after", f"{sum(p['status'] == 'ok' for p in after)}/{len(after)} packages ok",
            "done", time.time() - start)
    conflicts = pip_check(python, log) if install else ""

    info = {"manager": manager, "env_name": env_name, "prefix": str(prefix), "python": str(python),
            "created_or_checked": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_local_env(info)
    return {"info": info, "before": before, "after": after, "gpu": gpu, "pip_check": conflicts,
            "n_workers": n_workers}


def _project_installed(python: Path) -> bool:
    result = subprocess.run([str(python), "-c", "import t1dretrain"], capture_output=True, timeout=120,
                            cwd=str(Path.home()))
    return result.returncode == 0
