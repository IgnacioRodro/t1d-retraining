"""Stage 3 analysis (script 41; HANDOFF 7.5, D86, D90): cost, efficiency, forgetting and insulin per mechanism.

The unit is the patient, as in scripts 30 and 31 (skill pooled over the test
days against persistence on the same origins), with bootstrap intervals over
patients and the tie rule of D61 with the model's DMD (D75).

- **Efficiency**: the gain of an arm over never (mean over horizons, skill
  points) per computing second per test day; the reference arm (restart, D65)
  sets the scale.
- **Forgetting** (ARX): the skill of the update at every sample against its
  forgetting factor λ, with the effective memory 1 / (1 - λ) samples of 5 min.
- **Over the weeks**: the daily difference of each arm with the reference, week
  by week since the anchor (does a warm start drift away as it adds trees or
  training?).
- **Mechanism for the confirmation** (D91): per full model, the arm with the
  largest gain over never among those that beat the reference at one horizon
  at least and lose to it at none (tie rule with the model's DMD); otherwise
  the reference. A glucose-only twin takes the arm of its full model (D90).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..sim.mechanisms import parse_arm
from ..sim.stage0 import is_twin, split_variant
from . import policy_report as pr

STEP_MIN = 5.0


def effective_memory_days(forgetting: float) -> float:
    """About 1 / (1 - λ) samples of 5 min, in days (infinite without forgetting)."""
    return np.inf if forgetting >= 1 else STEP_MIN / (1 - forgetting) / 1440.0


def arm_table(skill: pd.DataFrame, fit_log: pd.DataFrame, test_days: int, references: dict) -> pd.DataFrame:
    """Per model and arm: mean skill per horizon and over horizons, the gain over never, computing seconds per test
    day and the efficiency (gain over never in points per computing second per day), and the cost and gain against
    the reference arm."""
    if skill.empty:
        return pd.DataFrame()
    table = pr.ranking(skill).drop(columns=["rank"])
    table = table.merge(pr.cost_table(fit_log, test_days), on=["period", "model", "policy"], how="left")
    by_horizon = skill.groupby(["period", "model", "policy", "horizon_min"])["skill"].mean().unstack("horizon_min")
    by_horizon.columns = [f"skill_{int(h)}" for h in by_horizon.columns]
    table = table.merge(by_horizon.reset_index(), on=["period", "model", "policy"], how="left")
    never = table[table["policy"] == "never"].set_index(["period", "model"])
    table["gain_over_never"] = [r.skill_mean - never["skill_mean"].get((r.period, r.model), np.nan)
                                for r in table.itertuples()]
    table["efficiency"] = 100 * table["gain_over_never"] / table["seconds_per_day"].where(table["seconds_per_day"] > 0)
    ref = table.set_index(["period", "model", "policy"])
    table["cost_vs_reference"] = [r.seconds_per_day / ref["seconds_per_day"].get((r.period, r.model,
                                                                                  references.get(r.model)), np.nan)
                                  for r in table.itertuples()]
    table["gain_vs_reference"] = [r.skill_mean - ref["skill_mean"].get((r.period, r.model, references.get(r.model)),
                                                                       np.nan) for r in table.itertuples()]
    table["kind"] = table["policy"].map(lambda a: parse_arm(a)[0])
    return table.rename(columns={"policy": "arm"}).sort_values(["period", "model", "skill_mean"],
                                                               ascending=[True, True, False]).reset_index(drop=True)


def forgetting_table(skill: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """ARX's online arms: per model, horizon and forgetting factor, the patient-mean skill, the effective memory and
    the paired difference with the reference arm (``paired``, from ``policy_report.paired``)."""
    s = skill[skill["policy"].str.startswith("online_l")]
    if s.empty:
        return pd.DataFrame()
    t = s.groupby(["period", "model", "policy", "horizon_min"])["skill"].mean().rename("skill_mean").reset_index()
    t["forgetting"] = t["policy"].map(lambda a: parse_arm(a)[1])
    t["memory_days"] = t["forgetting"].map(effective_memory_days)
    if not paired.empty:
        keys = ["period", "model", "policy", "horizon_min"]
        p = paired[keys + ["against", "skill_difference", "low", "high", "verdict"]]
        t = t.merge(p, on=keys, how="left")
    best = t.loc[t.groupby(["period", "model", "horizon_min"])["skill_mean"].idxmax()]
    t["best"] = t.index.isin(best.index)
    return t.sort_values(["period", "model", "horizon_min", "forgetting"]).reset_index(drop=True)


def confirmation_mechanisms(vs_reference: pd.DataFrame, table: pd.DataFrame, models: list[str], references: dict,
                            period: str) -> dict:
    """D91: the arm of each model for the confirmation (see the module docstring), with its reason and its verdicts
    against the reference per horizon; twins take their full model's arm."""
    out = {}
    v = vs_reference[vs_reference["period"] == period] if not vs_reference.empty else vs_reference
    t = table[table["period"] == period] if not table.empty else table
    for model in [m for m in models if not is_twin(m)]:
        reference = references.get(model, "restart")
        verdicts = {}
        if not v.empty:
            for arm, g in v[v["model"] == model].groupby("policy"):
                verdicts[arm] = {int(h): str(x) for h, x in zip(g["horizon_min"], g["verdict"])}
        gain = t[t["model"] == model].set_index("arm")["gain_over_never"] if not t.empty else pd.Series(dtype=float)
        qualified = [arm for arm, by in verdicts.items()
                     if "worse" not in by.values() and "better" in by.values() and arm in gain.index]
        arm = max(qualified, key=lambda a: gain[a]) if qualified else reference
        reason = (f"beats {reference} at {sum(x == 'better' for x in verdicts[arm].values())} of "
                  f"{len(verdicts[arm])} horizons and loses at none; largest gain over never among those"
                  if qualified else f"no arm beats {reference} without losing at a horizon: the reference")
        out[model] = {"arm": arm, "reference": reference, "reason": reason,
                      "gain_over_never_points": round(100 * float(gain.get(arm, np.nan)), 3),
                      "against_reference": verdicts.get(arm, {})}
    for model in [m for m in models if is_twin(m)]:
        full = split_variant(model)[0]
        if full in out:
            out[full]["twin"] = model
    return out
