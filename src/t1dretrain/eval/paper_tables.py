"""The tables of the paper (script 91, D105), built from the tables the other scripts wrote.

Nothing is recomputed: the functions select rows, turn skill fractions into
percentages and format them. The one derived number is the share of the daily
refit's gain that the selected interval keeps, a ratio of two gains that script
51 published (``gain_over_never`` and ``loss_to_daily`` of ``replication.csv``):
gain(k) / (gain(k) - difference(k, daily)).

Each table is returned as a tidy frame (written as CSV), then as text cells
ready for LaTeX (``*_cells``: free text escaped, numbers formatted) and as the
body of a LaTeX table (``latex_body``): a ``tabularx`` with booktabs rules.
Captions and notes stay in the manuscript, which wraps the body.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..sim.stage0 import is_twin

HORIZONS = (30, 60, 120)
NOT_APPLICABLE = "--"   # a cell that does not apply (for example, no daily refit outside REPLACE-BG)
PENDING = r"\pend"      # a value still to be added (macro of the manuscript)
NOT_RECORDED = "n.r."   # the dataset does not record it

MECHANISM_TEXT = {"restart": "Restart from the population model", "warm": "Continue training (warm start)"}
LATEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                 "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


# ============================================================================ formatting


def latex_escape(text) -> str:
    """Plain text (from the config or the data) made safe for LaTeX."""
    return "".join(LATEX_SPECIAL.get(c, c) for c in str(text))


def triplet(values, digits: int = 1) -> str:
    """One value per horizon: '22.2/19.2/19.5'."""
    return "/".join(f"{v:.{digits}f}" if np.isfinite(v) else "?" for v in values)


def span(low: float, high: float, digits: int = 0) -> str:
    """A range with an en dash, or one value when both ends round the same: '93--94', '93'."""
    a, b = f"{low:.{digits}f}", f"{high:.{digits}f}"
    return a if a == b else f"{a}--{b}"


def count(replicated, compared) -> str:
    """'32/33', or the not-applicable mark when nothing was compared."""
    if compared is None or (isinstance(compared, float) and not np.isfinite(compared)) or int(compared) == 0:
        return NOT_APPLICABLE
    return f"{int(replicated)}/{int(compared)}"


def latex_body(frame: pd.DataFrame, columns: list[tuple[str, str, str]], width: str = r"\linewidth") -> str:
    """A ``tabularx`` body from ``frame``, whose cells are already LaTeX (see the ``*_cells`` functions):
    ``columns`` lists (column of the frame, header, LaTeX alignment), where an alignment ``X`` is a ragged-right
    paragraph column."""
    spec = "".join(r">{\raggedright\arraybackslash}X" if align == "X" else align for _, _, align in columns)
    lines = [rf"\begin{{tabularx}}{{{width}}}{{@{{}}{spec}@{{}}}}", r"\toprule",
             " & ".join(header for _, header, _ in columns) + r"\\", r"\midrule"]
    for _, row in frame.iterrows():
        lines.append(" & ".join(str(row[name]) for name, _, _ in columns) + r"\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    return "\n".join(lines) + "\n"


# ============================================================================ the confirmation's plan


def split_arm(arm: str) -> tuple[str, int | None]:
    """('restart', 3) for 'restart_every3'; ('online_l1', None) for 'online_l1'."""
    match = re.fullmatch(r"(.+)_every(\d+)", str(arm))
    return (match.group(1), int(match.group(2))) if match else (str(arm), None)


def recommended_arms(replication: pd.DataFrame) -> pd.DataFrame:
    """Per full model, the interval and the mechanism the confirmation used (script 51, from scripts 31 and 41):
    the interval of its ``gain_over_never`` policy (``restart_every<k>``) and the arm of its
    ``mechanism_vs_restart`` rows (the restart when it has none)."""
    full = replication[~replication["model"].map(is_twin)]
    rows = []
    for model, part in full.groupby("model", sort=False):
        gain = part[part["finding"] == "gain_over_never"]["policy"].dropna().unique()
        mechanism = part[part["finding"] == "mechanism_vs_restart"]["policy"].dropna().unique()
        interval = split_arm(gain[0])[1] if len(gain) else None
        arm = split_arm(mechanism[0])[0] if len(mechanism) else "restart"
        rows.append({"model": model, "interval_days": interval, "arm": arm})
    return pd.DataFrame(rows)


def kept_share(replication: pd.DataFrame, cohort: str = "replace_bg/core") -> pd.DataFrame:
    """Per model and horizon, the share of the daily refit's gain that the selected interval keeps (%), from the
    cohort where the daily refit ran: gain(k) / (gain(k) - difference(k, daily))."""
    part = replication[replication["cohort"] == cohort]
    gain = part[part["finding"] == "gain_over_never"].set_index(["model", "horizon_min"])["skill_difference"]
    loss = part[part["finding"] == "loss_to_daily"].set_index(["model", "horizon_min"])["skill_difference"]
    both = pd.concat({"gain": gain, "loss": loss}, axis=1).dropna()
    both["share_pct"] = 100 * both["gain"] / (both["gain"] - both["loss"])
    return both.reset_index()


def mechanism_text(model: str, arm: str, finetune_train: str | None) -> str:
    """The mechanism in words. ``restart_other`` is the fine-tuning the pool did not choose (script 11): all the
    layers when it chose the head, the output layers otherwise."""
    if arm == "restart_other":
        return "Refit all layers" if finetune_train == "head" else "Refit the output layers only"
    if arm.startswith("online_l"):
        factor = arm.removeprefix("online_l")
        return ("Recursive update at every sample, no forgetting ($\\lambda=1$)" if float(factor) == 1
                else f"Recursive update at every sample, forgetting {factor}")
    return MECHANISM_TEXT.get(arm, arm)


# ============================================================================ table 1: datasets and cohorts


def cohort_table(rows: list[dict], cohort_summary: pd.DataFrame, cohorts: pd.DataFrame,
                 characteristics: pd.DataFrame | None, analysed: dict[str, int]) -> pd.DataFrame:
    """Table 1. ``rows`` (config ``paper_tables.cohorts``) gives the order and the descriptive text of every row;
    the counts come from script 04 (eligible patients, median usable days), script 06 (age, median and quartiles,
    and share of women among the eligible patients whose sex or gender is recorded) and ``analysed`` (patients
    each analysis used, by key). REPLACE-BG's groups have keys ``replace_bg/pool``, ``replace_bg/screen`` and
    ``replace_bg/confirm``. Without script 06 the age and the share of women are pending; a cohort whose dataset
    does not record them says so."""
    summary = cohort_summary.assign(key=cohort_summary["dataset"] + "/" + cohort_summary["cohort"]).set_index("key")
    core = cohorts[(cohorts["dataset"] == "replace_bg") & (cohorts["cohort"] == "core")]
    known = characteristics.set_index("key") if characteristics is not None else None
    out = []
    for row in rows:
        key = row["key"]
        dataset, cohort = key.split("/", 1)
        if dataset == "replace_bg" and cohort in ("pool", "screen", "confirm"):
            group = core[core["group"] == cohort]
            eligible, days = len(group), float(group["usable_days"].median())
        elif key in summary.index:
            eligible, days = int(summary.loc[key, "patients"]), float(summary.loc[key, "median_usable_days"])
        else:
            raise KeyError(f"paper_tables.cohorts: {key} is not a cohort of script 04")
        share, status = characteristic(known, key, "women_pct")
        age, age_status = characteristic(known, key, "age_median")
        q1, _ = characteristic(known, key, "age_q1")
        q3, _ = characteristic(known, key, "age_q3")
        out.append({"key": key, "dataset": row["dataset"], "cohort": row["cohort"], "therapy": row["therapy"],
                    "sensor": row["sensor"], "inputs": row["inputs"], "eligible": eligible,
                    "analysed": analysed.get(key), "median_days": days, "age_median": age, "age_q1": q1,
                    "age_q3": q3, "age_status": age_status, "female_pct": share, "female_status": status,
                    "role": row["role"]})
    return pd.DataFrame(out)


def characteristic(known: pd.DataFrame | None, key: str, column: str) -> tuple[float, str]:
    """A value of script 06 for a cohort and its status: recorded, not recorded (the dataset has no such datum)
    or pending (script 06 has not run for that cohort)."""
    if known is None or key not in known.index or column not in known.columns:
        return np.nan, "pending"
    value = float(known.loc[key, column])
    return value, "recorded" if np.isfinite(value) else "not recorded"


def cohort_cells(table: pd.DataFrame) -> pd.DataFrame:
    """Table 1 as text cells: an analysis that does not apply is '--'; a dataset that records no age, sex or gender
    is 'n.r.'; a value still to compute (script 06 not run) is pending."""
    cells = table.copy()
    for column in ("dataset", "cohort", "therapy", "sensor", "inputs", "role"):
        cells[column] = cells[column].map(latex_escape)
    cells["eligible"] = cells["eligible"].map(lambda v: f"{int(v)}")
    cells["analysed"] = cells["analysed"].map(lambda v: NOT_APPLICABLE if v is None or pd.isna(v) else f"{int(v)}")
    cells["therapy_sensor"] = cells["therapy"] + "; " + cells["sensor"]
    cells["median_days"] = cells["median_days"].map(lambda v: f"{v:.0f}")
    cells["age"] = [f"{r['age_median']:.0f} ({r['age_q1']:.0f}--{r['age_q3']:.0f})" if r["age_status"] == "recorded"
                    else NOT_RECORDED if r["age_status"] == "not recorded" else PENDING for _, r in table.iterrows()]
    cells["female_pct"] = [f"{v:.0f}" if status == "recorded" else NOT_RECORDED if status == "not recorded"
                           else PENDING for v, status in zip(table["female_pct"], table["female_status"])]
    return cells


# ============================================================================ table 2: models

FINETUNE_TEXT = {"head": "the output layers", "all": "all layers"}


def starting_point(family: str, chosen: dict) -> str:
    """The starting point of a learned model as the pool chose it (script 11, ``chosen_settings.yaml``): the size
    of the population model and what fine-tuning trains. Pending if script 11 has no population model for it."""
    settings = chosen.get(family) or {}
    population, finetune = settings.get("population") or {}, settings.get("finetune") or {}
    if not population:
        return PENDING
    if "num_leaves" in population:      # gradient-boosted trees
        size = f"{population['num_leaves']} leaves, learning rate {population['learning_rate']:g}"
        tuned = f"by adding trees (learning rate {finetune['learning_rate']:g})" if "learning_rate" in finetune else ""
    else:                               # networks
        layers, units = population.get("layers"), population.get("mlp_units")
        size = f"{layers} layers of {units} units" if layers else f"{units} units"
        tuned = f"on {FINETUNE_TEXT[finetune['train']]}" if finetune.get("train") in FINETUNE_TEXT else ""
    text = f"Population ({size})"
    return f"{text}; fine-tuned {tuned}" if tuned else text


def models_table(rows: list[dict], chosen: dict) -> pd.DataFrame:
    """Table 2. ``rows`` (config ``paper_tables.models``) gives the order and the descriptive text of every model;
    a row without ``start`` is a learned model whose starting point comes from the pool (``starting_point``)."""
    out = []
    for row in rows:
        start = row.get("start") or starting_point(row["family"], chosen)
        out.append({"family": row["family"], "label": row["label"], "kind": row["kind"], "inputs": row["inputs"],
                    "start": start, "refits": row["refits"], "stages": row["stages"]})
    return pd.DataFrame(out)


def models_cells(table: pd.DataFrame) -> pd.DataFrame:
    """Table 2 as text cells (a pending starting point keeps its mark)."""
    cells = table.copy()
    for column in ("label", "kind", "inputs", "refits", "stages"):
        cells[column] = cells[column].map(latex_escape)
    cells["start"] = [v if v == PENDING else latex_escape(v) for v in table["start"]]
    return cells


# ============================================================================ table 3: practical guide


def guide_table(skill: pd.DataFrame, replication: pd.DataFrame, cost: pd.DataFrame, models: list[str],
                finetune: dict[str, str | None], uses: dict[str, str], labels: dict[str, str],
                source: str = "screen (stage 3)") -> pd.DataFrame:
    """Table 3. Per model: interval and mechanism of the confirmation (script 51), skill frozen and refitted in
    ``source`` (script 74), share of the daily gain kept in REPLACE-BG (script 51), compute per test day of the
    refitted model in stage 3 (script 43, from the fits of script 40) and the recommended use (config)."""
    arms = recommended_arms(replication).set_index("model")
    kept = kept_share(replication)
    seconds = cost.set_index("forecaster")["seconds_per_day"]
    rows = []
    for model in models:
        if model not in arms.index:
            continue
        arm, interval = arms.loc[model, "arm"], arms.loc[model, "interval_days"]
        part = skill[skill["source"] == source]
        frozen = [float(part[(part["model"] == model) & (part["state"] == "frozen")
                             & (part["horizon_min"] == h)]["skill"].mean()) * 100 for h in HORIZONS]
        refitted = [float(part[(part["model"] == model) & (part["state"] == "refitted")
                               & (part["horizon_min"] == h)]["skill"].mean()) * 100 for h in HORIZONS]
        share = kept[kept["model"] == model]["share_pct"]
        rows.append({"model": model, "label": labels.get(model, model), "interval_days": interval,
                     "online": arm.startswith("online_l"), "arm": arm,
                     "mechanism": mechanism_text(model, arm, finetune.get(model)),
                     **{f"frozen_{h}": v for h, v in zip(HORIZONS, frozen)},
                     **{f"refitted_{h}": v for h, v in zip(HORIZONS, refitted)},
                     "kept_low": share.min() if len(share) else np.nan,
                     "kept_high": share.max() if len(share) else np.nan,
                     "seconds_per_day": float(seconds.get(f"{model} {arm}", np.nan)), "use": uses.get(model, "")})
    return pd.DataFrame(rows)


def guide_cells(table: pd.DataFrame) -> pd.DataFrame:
    """Table 3 as text cells. A model updated at every sample keeps its batch interval in the share column, marked
    with a footnote (a)."""
    cells = pd.DataFrame({"label": table["label"].map(latex_escape)})
    cells["interval"] = [r"Every sample\textsuperscript{a}" if online else f"{int(k)} days"
                         for online, k in zip(table["online"], table["interval_days"])]
    cells["mechanism"] = table["mechanism"]
    cells["frozen"] = [triplet([r[f"frozen_{h}"] for h in HORIZONS]) for _, r in table.iterrows()]
    cells["refitted"] = [triplet([r[f"refitted_{h}"] for h in HORIZONS]) for _, r in table.iterrows()]
    cells["kept"] = [span(r["kept_low"], r["kept_high"]) + (r"\textsuperscript{a}" if r["online"] else "")
                     if np.isfinite(r["kept_low"]) else PENDING for _, r in table.iterrows()]
    cells["seconds"] = [(f"{s:.2f}" if s < 0.1 else f"{s:.1f}") if np.isfinite(s) else PENDING
                        for s in table["seconds_per_day"]]
    cells["use"] = table["use"].map(latex_escape)
    return cells


# ============================================================================ table 4: replication


def replication_table(summary: pd.DataFrame, replication: pd.DataFrame, cohorts: list[str],
                      labels: dict[str, str]) -> pd.DataFrame:
    """Table 4. Per confirmation cohort (``cohorts``, in order): patients and, for each finding, the comparisons
    that replicated over those made (script 51, ``replication_summary.csv``; for the gain over frozen also the
    full models alone, from its ``full_models`` column)."""
    patients = replication.groupby("cohort")["patients"].max()
    out = []
    for cohort in cohorts:
        part = summary[summary["cohort"] == cohort].set_index("finding")
        if part.empty:
            continue
        row = {"cohort": cohort, "label": labels.get(cohort, cohort), "patients": int(patients.get(cohort, 0))}
        for finding in ("gain_over_never", "loss_to_daily", "mechanism_vs_restart"):
            if finding in part.index:
                row[f"{finding}_replicated"] = int(part.loc[finding, "replicated"])
                row[f"{finding}_compared"] = int(part.loc[finding, "comparisons"])
            else:
                row[f"{finding}_replicated"] = row[f"{finding}_compared"] = 0
        full = re.fullmatch(r"(\d+) of (\d+)", str(part.loc["gain_over_never", "full_models"]).strip()) \
            if "gain_over_never" in part.index else None
        row["full_replicated"], row["full_compared"] = (int(full.group(1)), int(full.group(2))) if full else (0, 0)
        out.append(row)
    return pd.DataFrame(out)


def replication_cells(table: pd.DataFrame) -> pd.DataFrame:
    cells = pd.DataFrame({"label": table["label"].map(latex_escape), "patients": table["patients"].map(str)})
    cells["gain"] = [count(r["gain_over_never_replicated"], r["gain_over_never_compared"]) for _, r in table.iterrows()]
    cells["full"] = [count(r["full_replicated"], r["full_compared"]) for _, r in table.iterrows()]
    cells["daily"] = [count(r["loss_to_daily_replicated"], r["loss_to_daily_compared"]) for _, r in table.iterrows()]
    cells["mechanism"] = [count(r["mechanism_vs_restart_replicated"], r["mechanism_vs_restart_compared"])
                          for _, r in table.iterrows()]
    return cells
