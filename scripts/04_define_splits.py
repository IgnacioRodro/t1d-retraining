"""04 - Cohorts, patient groups, day 0 and training anchors.

Reads the clean tables of script 03 and, for every dataset (rules in
src/t1dretrain/data/splits.py and decisions D12, D50-D52 of HANDOFF.md):

- counts the usable days of every patient by kind (open loop with complete
  insulin, open loop, closed loop with complete insulin, any valid day), in
  the whole follow-up and before / after randomisation;
- forms the cohorts (patients with >= 42 usable days of the kind needed);
- splits the REPLACE-BG core cohort into the groups pool / screen / confirm
  (seeded, stratified by arm and tertile of usable days, best balanced of
  many draws) and checks their balance; the groups are saved once in
  config/splits/replace_bg.yaml, later runs keep them (and report patients
  whose eligibility changed), and only --reassign redraws them;
- finds day 0 and checks which training anchors (days 14, 42, 70, 98) are
  available, for the whole follow-up and for the post-randomisation period;
- lists candidate real change events (randomisation, loop mode, CGM sensor).

Runs one job per dataset on all cores but two (rule R5).

Derived data:
    config/splits/<dataset>.yaml                 cohorts (and groups) per dataset
    <derived_data>/clean/<dataset>/anchors.parquet, cohorts.parquet, change_events.parquet

Outputs (results/04_define_splits/):
    usable_days.csv          per patient: usable days by kind and period
    cohorts.csv              per patient and cohort (with the group in the core cohort)
    cohort_summary.csv       per dataset and cohort: patients and usable days
    group_balance.csv        core groups: mean (SD) of each covariate and standardised mean differences
    anchor_availability.csv  per patient, cohort, period and anchor: day 0, training and test days, availability
    anchor_summary.csv       patients with each anchor available, by cohort, group and period
    change_events.csv        candidate real change events with valid days before and after
    group_balance.png
    anchor_availability.png
    followup_alignment.png
    cohort_sizes.png
    summary.md
    run_manifest.json

Usage:
    python scripts/04_define_splits.py
    python scripts/04_define_splits.py --reassign     # redraw the groups (only before any model was run)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain.envguard import ensure_project_env  # noqa: E402  (standard library only)

ensure_project_env(__file__)  # re-runs this script inside the project environment if needed

from t1dretrain.codecheck import verify_code  # noqa: E402  (standard library only)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from t1dretrain import __version__  # noqa: E402
from t1dretrain.compute import detect_resources, parallel_map  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.data.splits import (COHORTS, GROUP_ORDER, USABLE_RULES, add_strata, assess_dataset,  # noqa: E402
                                    assign_groups, balance_table)
from t1dretrain.data.store import ProcessedStore  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import (DATASET_COLORS, DATASET_LABELS, DATASET_ORDER, GROUP_COLORS,  # noqa: E402
                                 TEXT_SECONDARY, set_style)

SCRIPT_ID = "04_define_splits"
SPLITS_DIR = CONFIG_DIR / "splits"
COVARIATES = ["age", "female", "hba1c", "diabetes_duration_years", "cgm_only_arm", "usable_days",
              "usable_days_post", "randomization_offset", "mean_glucose", "glucose_cv", "tir_70_180",
              "carbs_g_per_day", "boluses_per_day"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--datasets", nargs="*", default=None, help="subset of datasets (default: all with clean data)")
    parser.add_argument("--reassign", action="store_true", help="redraw the core groups even if a split file exists")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()
    derived = derived_data_dir(args.paths)
    settings = dict(config["anchors"])
    anchors = [int(a) for a in config["stage1_anchor_days"]]
    min_valid = int(config.get("min_valid_days_core", 42))
    core = config.get("core_dataset", "replace_bg")
    seed = int(config["random_seed"])

    available = [d for d in DATASET_ORDER if (derived / "clean" / d / "days.parquet").exists()]
    datasets = [d for d in (args.datasets or available) if d in available]
    jobs = [{"dataset": d, "derived_root": derived, "settings": settings, "anchors": anchors,
             "min_valid_days": min_valid} for d in datasets]
    results = {r["dataset"]: r for r in parallel_map(assess_dataset, jobs, resources.n_workers, label="datasets")}

    counts = concat(r["counts"] for r in results.values())
    members = concat(r["members"] for r in results.values())
    anchor_table = concat(r["anchors"] for r in results.values())
    events = concat(r["events"] for r in results.values())

    groups = pd.DataFrame()
    balance = pd.DataFrame()
    draw, group_notes, saved_groups = {}, [], None
    if core in results and not members.empty:
        eligible = members[(members["dataset"] == core) & (members["cohort"] == "core")]
        eligible = core_covariates(results[core], counts, anchor_table, eligible)
        saved = load_saved_groups(core)
        if saved and not args.reassign:
            groups, draw, group_notes = reuse_groups(saved, eligible)
            saved_groups = saved["groups"]
        else:
            sizes = {**{g: int(n) for g, n in config["groups"].items()}, "confirm": None}
            groups, draw = assign_groups(eligible, sizes, seed, balance_on=config.get("group_balance_on"),
                                         draws=int(config.get("group_balance_draws", 1)))
            draw["written"] = date.today().isoformat()
            if saved:
                group_notes.append("groups redrawn with --reassign; the previous split file was replaced")
        balance = balance_table(groups.merge(eligible.drop(columns=["arm", "usable_days"]), on="patient_id",
                                             how="left"), COVARIATES)
        members = members.merge(groups[["patient_id", "group", "stratum"]].assign(dataset=core, cohort="core"),
                                on=["dataset", "patient_id", "cohort"], how="left")
    if "group" not in members:
        members["group"] = np.nan
    anchor_table = anchor_table.merge(members[["dataset", "patient_id", "cohort", "group"]],
                                      on=["dataset", "patient_id", "cohort"], how="left") \
        if not anchor_table.empty else anchor_table
    summary_cohorts = cohort_summary(members)
    summary_anchors = anchor_summary(anchor_table)

    for dataset in datasets:
        store = ProcessedStore(dataset, root=derived, stage="clean")
        store.write_extra("anchors", anchor_table[anchor_table["dataset"] == dataset])
        store.write_extra("cohorts", members[members["dataset"] == dataset])
        store.write_extra("change_events", events[events["dataset"] == dataset] if not events.empty else events)
        write_split_file(dataset, members[members["dataset"] == dataset], groups if dataset == core else None,
                         release, seed, min_valid, draw, saved_groups if dataset == core else None)

    out.csv(counts, "usable_days")
    out.csv(members, "cohorts")
    out.csv(summary_cohorts, "cohort_summary")
    out.csv(balance, "group_balance")
    out.csv(anchor_table, "anchor_availability")
    out.csv(summary_anchors, "anchor_summary")
    out.csv(events, "change_events")
    if not balance.empty:
        out.figure(plot_balance(balance), "group_balance")
    core_anchors = anchor_table[(anchor_table["dataset"] == core) & (anchor_table["cohort"] == "core")] \
        if not anchor_table.empty else anchor_table
    if not core_anchors.empty:
        out.figure(plot_anchor_availability(core_anchors, anchors, settings), "anchor_availability")
        out.figure(plot_alignment(derived, core, core_anchors, anchors), "followup_alignment")
    if not summary_cohorts.empty:
        out.figure(plot_cohorts(summary_cohorts), "cohort_sizes")
    out.text(write_summary(summary_cohorts, groups, draw, group_notes, balance, summary_anchors, core_anchors, events, settings,
                           config, release, resources, min_valid), "summary.md")
    out.manifest(settings={"datasets": datasets, "derived_data": str(derived), "anchors": anchors, **settings,
                           "groups": config["groups"], "seed": seed, "core_period": config.get("core_period")},
                 notes="One job per dataset; split files in config/splits/.")
    print(f"Done. Results in {out.dir}")


def concat(frames) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================================================================ groups


def load_saved_groups(dataset: str) -> dict:
    """Groups of an existing split file ({} if there is none)."""
    target = SPLITS_DIR / f"{dataset}.yaml"
    if not target.exists():
        return {}
    saved = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return saved if saved.get("groups") else {}


def reuse_groups(saved: dict, eligible: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[str]]:
    """The saved groups (D12: drawn once), restricted to the patients still eligible; notes on any change."""
    old = {str(pid): g for g, ids in saved["groups"].items() for pid in ids}
    print(f"Using the groups saved in config/splits/ ({len(old)} patients); --reassign redraws them")
    now = set(eligible["patient_id"].astype(str))
    notes = []
    gone = sorted(set(old) - now)
    new = sorted(now - set(old))
    if gone:
        notes.append(f"{len(gone)} patients of the saved groups are no longer eligible (they keep their group but "
                     f"have no core follow-up): {', '.join(gone[:10])}")
    if new:
        notes.append(f"{len(new)} newly eligible patients are in no group and are left out: {', '.join(new[:10])}")
    frame = add_strata(eligible[eligible["patient_id"].astype(str).isin(old)])  # strata as of this run
    frame["group"] = frame["patient_id"].astype(str).map(old)
    draw = dict(saved.get("group_draw") or {})
    draw.setdefault("written", saved.get("written"))
    return frame[["patient_id", "arm", "usable_days", "usable_days_tertile", "stratum", "group"]], draw, notes


def core_covariates(result: dict, counts: pd.DataFrame, anchors: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Covariates of the eligible core patients, used to balance the groups and to describe them."""
    frame = eligible[["patient_id", "usable_days"]].merge(counts[["patient_id", "arm"]], on="patient_id", how="left")
    frame = frame.merge(result["covariates"], on="patient_id", how="left")
    c = counts.set_index("patient_id")
    frame["usable_days_post"] = frame["patient_id"].map(c.get("usable_open_insulin_post", pd.Series(dtype=float)))
    frame["cgm_only_arm"] = (frame["arm"].astype(str).str.lower() == "cgm only").astype(float)
    whole = anchors[(anchors["cohort"] == "core") & (anchors["period"] == "whole")].drop_duplicates("patient_id")
    frame["randomization_offset"] = frame["patient_id"].map(whole.set_index("patient_id")["randomization_offset"])
    return frame


def write_split_file(dataset: str, members: pd.DataFrame, groups: pd.DataFrame | None, release: str, seed: int,
                     min_valid: int, draw: dict, saved_groups: dict | None = None) -> None:
    """Cohort lists and, for the core dataset, the groups (copied unchanged when they were reused)."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    content = {"dataset": dataset, "written": date.today().isoformat(), "release": release, "seed": seed,
               "usable_day_rules": USABLE_RULES, "min_usable_days": min_valid, "cohorts": {}}
    for cohort in COHORTS.get(dataset, []):
        ids = sorted(members.loc[members["cohort"] == cohort.name, "patient_id"].astype(str))
        content["cohorts"][cohort.name] = {"role": cohort.role, "description": cohort.description,
                                           "usable": cohort.usable, "min_usable_days": cohort.min_usable_days or
                                           min_valid, "arm": cohort.arm, "n": len(ids), "patients": ids}
    if groups is not None and not groups.empty:
        content["groups"] = saved_groups or {g: sorted(groups.loc[groups["group"] == g, "patient_id"].astype(str))
                                             for g in GROUP_ORDER}
        content["group_counts"] = {g: len(v) for g, v in content["groups"].items()}
        content["group_strata"] = "arm x tertile of usable open-loop days with complete insulin"
        content["group_draw"] = draw
    header = (f"# Written by scripts/04_define_splits.py (release {__version__}). Do not edit by hand: the groups are\n"
              "# drawn once (D12); later runs keep them, and only --reassign redraws them.\n")
    (SPLITS_DIR / f"{dataset}.yaml").write_text(header + yaml.safe_dump(content, sort_keys=False, allow_unicode=True),
                                                encoding="utf-8")


# ============================================================================ tables


def cohort_summary(members: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_ORDER:
        for cohort in COHORTS.get(dataset, []):
            m = members[(members["dataset"] == dataset) & (members["cohort"] == cohort.name)] \
                if not members.empty else members
            row = {"dataset": dataset, "cohort": cohort.name, "role": cohort.role, "description": cohort.description,
                   "usable_rule": cohort.usable, "patients": int(len(m)),
                   "usable_days": int(m["usable_days"].sum()) if len(m) else 0,
                   "median_usable_days": float(m["usable_days"].median()) if len(m) else np.nan}
            if len(m) and "group" in m and m["group"].notna().any():
                for g in GROUP_ORDER:
                    row[f"n_{g}"] = int((m["group"] == g).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def anchor_summary(anchors: pd.DataFrame) -> pd.DataFrame:
    if anchors.empty:
        return pd.DataFrame()
    frame = anchors.copy()
    frame["group"] = frame["group"].fillna("all")
    keys = ["dataset", "cohort", "group", "period", "anchor_day"]
    table = frame.groupby(keys, sort=False).agg(
        patients=("patient_id", "nunique"), available=("available", "sum"),
        median_test_days_within_max_age=("test_usable_days_within_max_age", "median"),
        median_days_after_day0=("days_after_day0", "median")).reset_index()
    table["available_share"] = (table["available"] / table["patients"]).round(3)
    core = frame[frame["group"] != "all"].assign(group="all")  # core cohort pooled over its groups
    if not core.empty:
        pooled = core.groupby(keys, sort=False).agg(
            patients=("patient_id", "nunique"), available=("available", "sum"),
            median_test_days_within_max_age=("test_usable_days_within_max_age", "median"),
            median_days_after_day0=("days_after_day0", "median")).reset_index()
        pooled["available_share"] = (pooled["available"] / pooled["patients"]).round(3)
        table = pd.concat([table, pooled], ignore_index=True)
    table["dataset_order"] = table["dataset"].map(DATASET_ORDER.index)
    table["group_order"] = table["group"].map({g: k for k, g in enumerate(GROUP_ORDER + ["all"])})
    return table.sort_values(["dataset_order", "cohort", "period", "group_order", "anchor_day"]).drop(
        columns=["dataset_order", "group_order"]).reset_index(drop=True)


# ============================================================================ figures


def plot_balance(balance: pd.DataFrame) -> plt.Figure:
    pairs = [c for c in balance.columns if c.startswith("smd_")]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(balance) + 1.6))
    y = np.arange(len(balance))[::-1]
    markers = ["o", "s", "^"]
    for k, pair in enumerate(pairs):
        g1, g2 = pair[4:].split("_vs_")
        ax.scatter(balance[pair], y, marker=markers[k % 3], s=36, color=GROUP_COLORS.get(g2, TEXT_SECONDARY),
                   edgecolors=GROUP_COLORS.get(g1, TEXT_SECONDARY), linewidths=1.4, label=f"{g1} vs {g2}", zorder=3)
    for x in (-0.1, 0.1):
        ax.axvline(x, color=TEXT_SECONDARY, lw=0.8, ls="--")
    ax.axvline(0, color=TEXT_SECONDARY, lw=0.6)
    ax.set_yticks(y, balance["covariate"])
    ax.set_xlabel("standardised mean difference")
    ax.set_title("Balance of the core groups (dashed: |SMD| = 0.1)", loc="left")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_anchor_availability(anchors: pd.DataFrame, anchor_days: list[int], settings: dict) -> plt.Figure:
    periods = [p for p in ["whole", "post_randomization"] if p in set(anchors["period"])]
    fig, axes = plt.subplots(1, len(periods), figsize=(6.2 * len(periods), 4.2), sharey=True, squeeze=False)
    width = 0.26
    for ax, period in zip(axes[0], periods):
        a = anchors[anchors["period"] == period]
        for k, group in enumerate(GROUP_ORDER):
            g = a[a["group"] == group]
            n = g["patient_id"].nunique()
            share = [100 * g.loc[g["anchor_day"] == d, "available"].mean() if n else 0 for d in anchor_days]
            x = np.arange(len(anchor_days)) + (k - 1) * width
            ax.bar(x, share, width=width, color=GROUP_COLORS[group], label=f"{group} (n={n})")
            for xx, s in zip(x, share):
                ax.text(xx, s + 1, f"{s:.0f}", ha="center", fontsize=7, color=TEXT_SECONDARY)
        ax.set_xticks(np.arange(len(anchor_days)), [f"day {d}" for d in anchor_days])
        ax.set_ylim(0, 110)
        ax.set_title(f"{period.replace('_', ' ')} follow-up", loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=3, fontsize=8)
    axes[0, 0].set_ylabel("patients with the anchor available (%)")
    fig.suptitle(f"Training anchors: >= {settings['min_train_usable_days']} usable days in the "
                 f"{settings['train_window_days']} before, >= {settings['min_test_usable_days']} in the "
                 f"{settings['max_age_days']} after", fontweight="bold", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def plot_alignment(derived: Path, core: str, anchors: pd.DataFrame, anchor_days: list[int]) -> plt.Figure:
    """Follow-up of each core patient aligned at day 0 of the whole period, by group."""
    from t1dretrain.data.splits import usable_flags

    days = pd.read_parquet(derived / "clean" / core / "days.parquet")
    days = pd.concat([days, usable_flags(days)], axis=1)
    whole = anchors[anchors["period"] == "whole"].drop_duplicates("patient_id").dropna(subset=["day0_index"])
    whole = whole.assign(group_order=whole["group"].map({g: k for k, g in enumerate(GROUP_ORDER)}))
    whole = whole.sort_values(["group_order", "randomization_offset"])
    lo, hi = -60, 300
    grid = np.full((len(whole), hi - lo), 3)
    marks = []
    by_patient = dict(tuple(days.groupby("patient_id")))
    for i, (_, row) in enumerate(whole.iterrows()):
        p = by_patient[row["patient_id"]]
        x = p["day_index"].to_numpy() - int(row["day0_index"]) - lo
        code = np.where(p["usable_open_insulin"], 0, np.where(p["valid"], 1, np.where(p["n_readings"] > 0, 2, 3)))
        ok = (x >= 0) & (x < hi - lo)
        grid[i, x[ok]] = code[ok]
        if np.isfinite(row["randomization_offset"]):
            marks.append((i, row["randomization_offset"]))
    colors = ["#2a78d6", "#9ec5f4", "#c9c8c3", "#ffffff"]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(grid, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(colors), vmin=-0.5, vmax=3.5,
              interpolation="nearest", extent=(lo, hi, len(whole), 0))
    if marks:
        rows, offsets = zip(*marks)
        ax.scatter(offsets, np.array(rows) + 0.5, marker="|", s=10, color="#0b0b0b", linewidths=0.8,
                   label="randomisation")
    for d in anchor_days:
        ax.axvline(d, color="#eb6834", lw=0.9, ls="--")
    start = 0
    for group in GROUP_ORDER:
        n = int((whole["group"] == group).sum())
        if n:
            ax.axhline(start + n, color="#0b0b0b", lw=0.8)
            ax.text(hi + 3, start + n / 2, f"{group}\n(n={n})", va="center", fontsize=8)
        start += n
    ax.set_xlabel("days since day 0 (whole follow-up); dashed: training anchors")
    ax.set_ylabel("core patients")
    ax.set_yticks([])
    ax.grid(False)
    handles = [plt.matplotlib.patches.Patch(facecolor=c, edgecolor="#9b9a96", label=label) for c, label in
               zip(colors, ["usable (open loop, insulin complete)", "valid CGM only", "too few readings",
                            "no readings"])]
    handles.append(plt.matplotlib.lines.Line2D([], [], marker="|", ls="", color="#0b0b0b", label="randomisation"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=5, fontsize=8)
    ax.set_title(f"{DATASET_LABELS.get(core, core)}: follow-up aligned at day 0, by group", loc="left")
    fig.tight_layout()
    return fig


def plot_cohorts(summary: pd.DataFrame) -> plt.Figure:
    s = summary[summary["patients"] > 0].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(s) + 1.4))
    y = np.arange(len(s))[::-1]
    ax.barh(y, s["patients"], color=[DATASET_COLORS[d] for d in s["dataset"]], height=0.7)
    for yy, (_, row) in zip(y, s.iterrows()):
        ax.text(row["patients"], yy, f" {row['patients']} patients, {row['usable_days']:,} usable days", va="center",
                fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(y, [f"{DATASET_LABELS[d]} · {c}" for d, c in zip(s["dataset"], s["cohort"])])
    ax.set_xscale("log")
    ax.set_xlim(1, s["patients"].max() * 12)
    ax.set_xlabel("patients (log scale)")
    ax.set_title("Cohorts (>= the minimum of usable days of the kind each needs)", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(cohorts: pd.DataFrame, groups: pd.DataFrame, draw: dict, group_notes: list[str],
                  balance: pd.DataFrame, anchors: pd.DataFrame,
                  core_anchors: pd.DataFrame, events: pd.DataFrame, settings: dict, config: dict, release: str,
                  resources, min_valid: int) -> str:
    lines = ["# 04 - Cohorts, groups and anchors", "", f"Code release {release}.", f"Machine: {resources.describe()}",
             f"Cohorts need >= {min_valid} usable days of their kind; anchors {config['stage1_anchor_days']} need >= "
             f"{settings['min_train_usable_days']} usable days in the {settings['train_window_days']} before and >= "
             f"{settings['min_test_usable_days']} in the {settings['max_age_days']} after. Core period (provisional): "
             f"`{config.get('core_period')}`.", ""]
    problems = list(group_notes)
    if not balance.empty:
        smd_cols = [c for c in balance.columns if c.startswith("smd_")]
        worst = balance.set_index("covariate")[smd_cols].abs().max(axis=1)
        for covariate, value in worst[worst > 0.2].items():
            problems.append(f"core groups: |SMD| = {value:.2f} for {covariate}")
    lines += ["## Warnings", ""] + ([f"- {p}" for p in problems] or ["- none"]) + [""]
    if not cohorts.empty:
        table = cohorts.assign(dataset=cohorts["dataset"].map(DATASET_LABELS)).drop(columns=["description"])
        lines += ["## Cohorts", "", markdown_table(table), ""]
    if not groups.empty:
        lines += ["## Core groups by stratum", "",
                  f"Drawn on {draw.get('written', '?')}: best of {draw.get('draws')} stratified draws (draw "
                  f"{draw.get('chosen_draw')}), largest |SMD| {draw.get('max_abs_smd')} over "
                  f"{', '.join(draw.get('balance_on', []))}. Strata as of this run.", "",
                  markdown_table(groups.pivot_table(index="stratum", columns="group", values="patient_id",
                                                    aggfunc="count", fill_value=0)[GROUP_ORDER].reset_index()), ""]
    if not balance.empty:
        lines += ["## Balance of the core groups", "", markdown_table(balance), ""]
    if not anchors.empty:
        core = anchors[anchors["group"] == "all"]
        lines += ["## Anchors available (all groups pooled)", "",
                  markdown_table(core.assign(dataset=core["dataset"].map(DATASET_LABELS))), ""]
    if not core_anchors.empty:
        whole = core_anchors[core_anchors["period"] == "whole"].drop_duplicates("patient_id")
        offsets = whole["randomization_offset"].dropna()
        if len(offsets):
            lines += ["## Randomisation in the core cohort (whole follow-up)", "",
                      f"Days from day 0 to randomisation: median {offsets.median():.0f} (IQR {offsets.quantile(0.25):.0f}"
                      f"-{offsets.quantile(0.75):.0f}, range {offsets.min():.0f} to {offsets.max():.0f}). Anchors that "
                      "fall after it train on pre-randomisation days and test after the change (D52).", ""]
    if not events.empty:
        table = events.groupby(["dataset", "kind"]).agg(
            patients=("patient_id", "nunique"), events=("patient_id", "size"),
            median_valid_days_before=("valid_days_before", "median"),
            median_valid_days_after=("valid_days_after", "median")).reset_index()
        lines += ["## Candidate real change events", "",
                  markdown_table(table.assign(dataset=table["dataset"].map(DATASET_LABELS))), ""]
    lines += ["## Checks to look at", "",
              "- `group_balance.png`: all |SMD| near or below 0.1.",
              "- `anchor_availability.png`: how many patients keep each anchor in each period.",
              "- `followup_alignment.png`: where randomisation and the gaps fall relative to the anchors.",
              "- `change_events.csv`: candidate changes for the recovery analysis (stage 3).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
