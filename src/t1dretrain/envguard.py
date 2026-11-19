"""Make every script run inside the project environment.

Scripts call ``ensure_project_env(__file__)`` before importing any third-party
package. If they were launched with another Python, the guard re-runs them with
the interpreter recorded by script 00 in ``config/environment.local.json`` and
exits with its return code. Standard library only.

Set ``T1D_SKIP_ENV_GUARD=1`` to disable the guard (for example inside a
container that already is the environment).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .envsetup import LOCAL_ENV_FILE, read_local_env


def running_in(prefix: str) -> bool:
    """True when the current interpreter belongs to the environment at ``prefix``."""
    try:
        return Path(sys.prefix).resolve() == Path(prefix).resolve()
    except OSError:
        return False


def ensure_project_env(script: str) -> None:
    """Re-run ``script`` inside the project environment when needed."""
    if os.environ.get("T1D_SKIP_ENV_GUARD") == "1":
        return
    info = read_local_env()
    if info is None:
        sys.exit(f"The project environment is not set up ({LOCAL_ENV_FILE} missing). "
                 "Run: python scripts/00_check_environment.py")
    if running_in(info["prefix"]):
        return
    if os.environ.get("T1D_ENV_ACTIVE") == "1":
        print(f"WARNING: expected the environment at {info['prefix']} but running {sys.prefix}; continuing.")
        return
    python = info["python"]
    if not Path(python).exists():
        sys.exit(f"The project environment recorded in {LOCAL_ENV_FILE} no longer exists ({python}). "
                 "Run: python scripts/00_check_environment.py")
    print(f"Switching to the project environment: {python}")
    env = dict(os.environ, T1D_ENV_ACTIVE="1")
    result = subprocess.run([python, str(Path(script).resolve()), *sys.argv[1:]], env=env)
    sys.exit(result.returncode)
