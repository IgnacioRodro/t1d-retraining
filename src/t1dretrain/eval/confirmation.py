"""Confirmation (scripts 50 and 51; HANDOFF 7.6, D92): what runs on each cohort, and the measures it adds.

Nothing is chosen in the confirmation. Each model brings what the screening
chose for it: the policies of script 31 (``confirmation.yaml``, D84: never, the
growing window at the optimal interval ``k`` and every day; ARX also online)
and the mechanism of script 41 (``mechanisms.yaml``, D91). As arms of
``sim.mechanisms``:

- ``never``;
- ``restart_every<k>`` and ``restart_every1``: the growing window, each refit
  from the population model (D65), the same fits as stage 2's
  ``growing_every<k>`` (stage 3's ``restart`` is identical to them);
- ``online_l1`` (ARX): the update at every sample without forgetting (D87);
- ``<mechanism>_every<k>``: the mechanism of script 41 at the optimal interval,
  when it is not the restart (for example ``warm_every3``).

A glucose-only twin runs the arms of its full model (D90). Forecast lows: for
every threshold (70 and 54 mg/dL), the sensitivity (share of the reference
values below it that the forecast puts below it) and the precision (share of
the forecasts below it whose reference is below it), pooled over forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..sim.mechanisms import parse_arm, split_interval
from ..sim.stage0 import is_twin, needs_exog, split_variant

INSULIN_RULES = ("open_insulin", "closed_insulin", "insulin")  # usable-day rules with complete insulin (data.splits)


def cohort_key(dataset: str, cohort: str) -> str:
    return f"{dataset}/{cohort}"


def reads_insulin(model: str) -> bool:
    """Whether a model needs insulin and carbohydrates (the full learned models and ARX; not AR(3) or a twin)."""
    return needs_exog(model) and not is_twin(model)


def models_for_rule(models: list[str], rule: str) -> list[str]:
    """The models that can run on a cohort: those that read the glucose only when the cohort has no insulin."""
    return list(models) if rule in INSULIN_RULES else [m for m in models if not reads_insulin(m)]


def confirmation_arms(policies: list[str], mechanism: str | None, interval: int | None, daily: bool,
                      forgetting: float = 1.0) -> list[str]:
    """The arms of one model (see the module docstring) from its policies of script 31 and its mechanism of script
    41; without ``daily`` the every-day refit is left out (unless never is enough and it is the only refit)."""
    arms = []
    for policy in policies:
        if policy == "never":
            arms.append("never")
        elif policy == "online":
            arms.append(f"online_l{float(forgetting):g}")
        elif policy.startswith("growing_every"):
            every = int(policy[len("growing_every"):])
            if every == 1 and not daily and interval:
                continue
            arms.append(f"restart_every{every}")
        else:
            raise ValueError(f"policy {policy!r} has no arm in the confirmation")
    if mechanism:
        if parse_arm(mechanism)[0] == "online":
            arms.append(mechanism)
        elif mechanism != "restart" and interval:
            arms.append(f"{mechanism}_every{int(interval)}")
    return list(dict.fromkeys(arms))


def mechanism_arm(entry: dict) -> str | None:
    """The confirmation arm of a model's mechanism (``entry``: its plan, with ``mechanism`` and ``interval``):
    ``<arm>_every<k>``, or ARX's online update; None for the restart (the comparison is then the restart itself)."""
    mechanism, k = entry.get("mechanism"), entry.get("interval")
    if not mechanism or mechanism == "restart":
        return None
    if mechanism.startswith("online_l"):
        return mechanism
    return f"{mechanism}_every{int(k)}" if k else None


def recommended_arm(entry: dict) -> str | None:
    """What the study recommends for a model, as a confirmation arm: its mechanism at its optimal interval, or the
    restart at that interval (``restart_every<k>``); None when the model has no interval (never is enough)."""
    k = entry.get("interval")
    return mechanism_arm(entry) or (f"restart_every{int(k)}" if k else None)


def screen_names(arm: str, interval: int | None) -> tuple[str | None, str | None]:
    """The names of a confirmation arm in the screening: (policy of stage 2, arm of stage 3). ``restart_every<k>``
    is stage 2's ``growing_every<k>`` (and stage 3's ``restart`` when ``k`` is the model's interval);
    ``warm_every<k>`` is stage 3's ``warm``; never and the online update keep their names."""
    base, every = split_interval(arm)
    if every is None:
        return ("online" if base.startswith("online_l") else base), base
    stage2 = f"growing_every{every}" if base == "restart" else None
    stage3 = base if every == interval else None
    return stage2, stage3


def full_model(model: str) -> str:
    return split_variant(model)[0]


def low_columns(frame: pd.DataFrame) -> list[int]:
    """The thresholds whose counts a day-metrics frame holds (``below<t>_true``, ...)."""
    out = []
    for column in frame.columns:
        if column.startswith("below") and column.endswith("_true"):
            out.append(int(float(column[len("below"):-len("_true")])))
    return sorted(set(out), reverse=True)


def low_sums(frame: pd.DataFrame) -> list[str]:
    return [f"below{t}_{kind}" for t in low_columns(frame) for kind in ("true", "alarm", "hit")]


def with_lows(table: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity and precision of the forecast lows from the pooled counts, in %, replacing the counts."""
    table = table.copy()
    for t in low_columns(table):
        true, alarm, hit = (table[f"below{t}_{k}"] for k in ("true", "alarm", "hit"))
        table[f"low{t}_sensitivity_pct"] = 100 * hit / true.where(true > 0)
        table[f"low{t}_precision_pct"] = 100 * hit / alarm.where(alarm > 0)
        table[f"low{t}_share_pct"] = 100 * true / table["n"].where(table["n"] > 0)
        table = table.drop(columns=[f"below{t}_{k}" for k in ("true", "alarm", "hit")])
    return table


def draw_patients(patients: list[str], n: int | None, seed: int, key: str) -> list[str]:
    """``n`` patients drawn at random (sorted), the same in every run: the project seed and the cohort's name."""
    import zlib

    patients = sorted(patients)
    if not n or n >= len(patients):
        return patients
    rng = np.random.default_rng([int(seed), zlib.crc32(key.encode())])
    return sorted(rng.choice(patients, size=int(n), replace=False).tolist())
