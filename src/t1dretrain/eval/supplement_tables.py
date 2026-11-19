"""The tables of the supplementary material (script 92, D111), built from the tables of scripts 51, 61 and 62.

As for the paper's tables (``eval.paper_tables``), nothing is recomputed: the
functions select rows, orient and format them. Each table is a tidy frame
(written as CSV) and text cells ready for LaTeX (``*_cells``), which
``paper_tables.latex_body`` turns into the body of a ``tabularx``.

    regularity_table   S1: the primary regularity indices against the four per-patient outcomes (script 61)
    sex_counts_table   S2: women, men, other genders and not recorded among the patients with outcomes (script 62)
    error_table        S3: RMSE (mg/dL) frozen and refitted and Clarke zones of the refitted model (script 51)
    lows_table         S4: sensitivity and precision of the forecast lows, frozen and refitted (script 51)

"Refitted" is the arm the confirmation recommends for each model: its selected
mechanism at its selected interval, or the restart at that interval when the
restart was kept (``refit_policy``); a glucose-only twin has its full model's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..sim.stage0 import TWIN, is_twin
from .paper_tables import HORIZONS, NOT_APPLICABLE, latex_escape, triplet

FROZEN = "never"
REFERENCE = "reference"   # the arm of persistence


# ============================================================================ helpers


def signed(value: float, digits: int = 2) -> str:
    """'+0.22' or a true minus sign ('\\textminus{}0.03') for LaTeX."""
    return f"{value:+.{digits}f}".replace("-", r"\textminus{}")


def refit_policy(replication: pd.DataFrame, model: str) -> str | None:
    """The refitted arm of ``model`` in the confirmation (script 51): the arm of its ``mechanism_vs_restart`` rows
    (the mechanism the screening selected) or, if it has none, that of its ``gain_over_never`` rows (the restart at
    the selected interval)."""
    part = replication[replication["model"] == model]
    for finding in ("mechanism_vs_restart", "gain_over_never"):
        policy = part[part["finding"] == finding]["policy"].dropna().unique()
        if len(policy):
            return str(policy[0])
    return None


def cohort_models(table: pd.DataFrame, cohort: str, models: list[str]) -> list[str]:
    """The models of ``cohort`` in the order of ``models`` (full models): a full model where it ran, otherwise its
    glucose-only twin (cohorts without insulin)."""
    present = set(table[table["cohort"] == cohort]["model"])
    out = []
    for model in models:
        if model in present:
            out.append(model)
        elif f"{model}_{TWIN}" in present:
            out.append(f"{model}_{TWIN}")
    return out


def model_label(model: str, labels: dict[str, str]) -> str:
    base = model.removesuffix(f"_{TWIN}")
    text = labels.get(base, base)
    return f"{text}, glucose only" if is_twin(model) else text


def first_only(values: list[str]) -> list[str]:
    """A grouping column printed on the first row of each group only."""
    return ["" if i and v == values[i - 1] else v for i, v in enumerate(values)]


# ============================================================================ S1: regularity (script 61)


def regularity_table(associations: pd.DataFrame, models: list[str], outcomes: list[str],
                     indices: list[str]) -> pd.DataFrame:
    """S1. For the primary indices and the full ``models``, the coefficient of each outcome (mean over the horizons)
    in skill points per standard deviation towards the more regular side, with its 95 % interval turned the same way
    (script 61 gives the interval of the raw coefficient), the Holm p and the patients."""
    part = associations[(associations["horizon_min"] == 0) & associations["index"].isin(indices)
                        & associations["model"].isin(models) & associations["outcome"].isin(outcomes)]
    rows = []
    for _, r in part.iterrows():
        flip = r.get("regular_when") == "lower"
        low, high = (-r["high"], -r["low"]) if flip else (r["low"], r["high"])
        coef = r["coef_regular"] if pd.notna(r.get("coef_regular")) else r["coef"]
        rows.append({"index": r["index"], "model": r["model"], "outcome": r["outcome"], "coef": float(coef),
                     "low": float(low), "high": float(high), "p_holm": float(r["p_holm"]),
                     "patients": int(r["patients"])})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    order = {"index": {k: i for i, k in enumerate(indices)}, "model": {k: i for i, k in enumerate(models)}}
    return (table.assign(_i=table["index"].map(order["index"]), _m=table["model"].map(order["model"]))
            .sort_values(["_i", "_m"]).drop(columns=["_i", "_m"]).reset_index(drop=True))


def regularity_cells(table: pd.DataFrame, outcomes: list[str], index_labels: dict[str, str],
                     labels: dict[str, str]) -> pd.DataFrame:
    """S1 as cells: one row per index and model, one column per outcome ('+0.02 (\\textminus{}0.03, +0.07)')."""
    rows = []
    for (index, model), part in table.groupby(["index", "model"], sort=False):
        row = {"index": latex_escape(index_labels.get(index, index)), "model": latex_escape(model_label(model, labels))}
        for outcome in outcomes:
            hit = part[part["outcome"] == outcome]
            row[outcome] = (f"{signed(hit['coef'].iloc[0])} ({signed(hit['low'].iloc[0])}, "
                            f"{signed(hit['high'].iloc[0])})") if len(hit) else NOT_APPLICABLE
        rows.append(row)
    cells = pd.DataFrame(rows)
    if not cells.empty:
        cells["index"] = first_only(cells["index"].tolist())
    return cells


# ============================================================================ S2: sex (script 62)


def sex_counts_table(counts: pd.DataFrame, cohort_labels: dict[str, str],
                     order: list[str] | None = None) -> pd.DataFrame:
    """S2. Per cohort (in ``order``, then any other), the patients with outcomes by sex or gender, and a total row."""
    columns = ["women", "men", "other", "not recorded"]
    table = counts.copy()
    if order:
        rank = {c: i for i, c in enumerate(order)}
        table = table.assign(_r=table["cohort"].map(lambda c: rank.get(c, len(rank)))).sort_values(
            ["_r", "cohort"]).drop(columns="_r")
    for column in columns:
        if column not in table:
            table[column] = 0
    table = table[["cohort"] + columns]
    table.insert(1, "label", table["cohort"].map(lambda c: cohort_labels.get(c, c)))
    total = {"cohort": "total", "label": "Total", **{c: int(table[c].sum()) for c in columns}}
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


def sex_counts_cells(table: pd.DataFrame) -> pd.DataFrame:
    cells = pd.DataFrame({"label": table["label"].map(latex_escape)})
    for column in ("women", "men", "other", "not recorded"):
        cells[column] = table[column].map(lambda v: f"{int(v)}")
    return cells


# ============================================================================ S3: error and Clarke zones (script 51)


def error_table(metrics: pd.DataFrame, replication: pd.DataFrame, cohorts: list[str],
                models: list[str]) -> pd.DataFrame:
    """S3. Per cohort and model: RMSE (mg/dL) frozen and refitted, and the share of forecasts of the refitted model
    in zones A and B and in zone D of the Clarke error grid, per horizon (``metrics_by_arm.csv`` of script 51).
    Persistence has one arm (``reference``), shown as frozen."""
    rows = []
    for cohort in cohorts:
        part = metrics[metrics["cohort"] == cohort]
        for model in cohort_models(metrics, cohort, models):
            m = part[part["model"] == model]
            frozen = REFERENCE if model == "persistence" else FROZEN
            refit = None if model == "persistence" else refit_policy(replication, model)
            row = {"cohort": cohort, "model": model, "frozen_policy": frozen, "refit_policy": refit}
            for h in HORIZONS:
                f = m[(m["policy"] == frozen) & (m["horizon_min"] == h)]
                r = m[(m["policy"] == refit) & (m["horizon_min"] == h)] if refit else m.iloc[0:0]
                row[f"rmse_frozen_{h}"] = float(f["rmse"].iloc[0]) if len(f) else np.nan
                row[f"rmse_refit_{h}"] = float(r["rmse"].iloc[0]) if len(r) else np.nan
                source = r if len(r) else f   # persistence: its own zones
                row[f"ab_{h}"] = float(source["clarke_AB_pct"].iloc[0]) if len(source) else np.nan
                row[f"d_{h}"] = float(source["clarke_D_pct"].iloc[0]) if len(source) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def error_cells(table: pd.DataFrame, cohort_labels: dict[str, str], labels: dict[str, str]) -> pd.DataFrame:
    def values(r, key, digits=1):
        vals = [r[f"{key}_{h}"] for h in HORIZONS]
        return triplet(vals, digits) if np.isfinite(vals).any() else NOT_APPLICABLE

    cells = pd.DataFrame({"cohort": first_only([latex_escape(cohort_labels.get(c, c)) for c in table["cohort"]]),
                          "model": [latex_escape(model_label(m, labels)) for m in table["model"]]})
    cells["rmse_frozen"] = [values(r, "rmse_frozen") for _, r in table.iterrows()]
    cells["rmse_refit"] = [values(r, "rmse_refit") for _, r in table.iterrows()]
    cells["ab"] = [values(r, "ab") for _, r in table.iterrows()]
    cells["d"] = [values(r, "d") for _, r in table.iterrows()]
    return cells


# ============================================================================ S4: hypoglycemia (script 51)


def lows_table(lows: pd.DataFrame, replication: pd.DataFrame, cohorts: list[str], models: list[str],
               horizons: tuple[int, ...] = (30, 60), threshold: int = 70) -> pd.DataFrame:
    """S4. Per cohort, model and state (frozen, refitted; persistence: reference), sensitivity and precision (%) of
    the forecast lows below ``threshold`` mg/dL with the alarm at the same level (``low_glucose.csv`` of script 51),
    and the share of targets below it."""
    rows = []
    for cohort in cohorts:
        part = lows[lows["cohort"] == cohort]
        for model in cohort_models(lows, cohort, models):
            states = [("reference", REFERENCE)] if model == "persistence" else \
                [("frozen", FROZEN), ("refitted", refit_policy(replication, model))]
            for state, policy in states:
                m = part[(part["model"] == model) & (part["policy"] == policy)]
                row = {"cohort": cohort, "model": model, "state": state, "policy": policy}
                for h in horizons:
                    hit = m[m["horizon_min"] == h]
                    for name in ("sensitivity", "precision", "share"):
                        column = f"low{threshold}_{name}_pct"
                        row[f"{name}_{h}"] = float(hit[column].iloc[0]) if len(hit) and column in hit else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def lows_cells(table: pd.DataFrame, cohort_labels: dict[str, str], labels: dict[str, str],
               horizons: tuple[int, ...] = (30, 60)) -> pd.DataFrame:
    def pct(v):
        return f"{v:.0f}" if np.isfinite(v) else NOT_APPLICABLE

    share = table.groupby("cohort")[f"share_{horizons[0]}"].first()
    cohort = [latex_escape(cohort_labels.get(c, c)) + (f" ({share[c]:.1f}\\%)" if np.isfinite(share[c]) else "")
              for c in table["cohort"]]
    cells = pd.DataFrame({"cohort": first_only(cohort),
                          "model": [latex_escape(model_label(m, labels)) for m in table["model"]],
                          "state": table["state"].str.capitalize()})
    for h in horizons:
        cells[f"sensitivity_{h}"] = table[f"sensitivity_{h}"].map(pct)
        cells[f"precision_{h}"] = table[f"precision_{h}"].map(pct)
    return cells
