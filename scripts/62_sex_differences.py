"""62 - Do the outcomes of the confirmation differ between women and men? (SAGER; HANDOFF 2, D109)

A brief check for the supplement of the paper: the SAGER guidelines ask whether
the results were analysed by sex. Same patients, outcomes and adjustment as the
regularity analysis (script 61; ``eval.sex_differences``): per model, its
glucose-only twin and their difference, the skill of the frozen model, how fast
it ages, the gain of refitting at the optimal interval and what that interval
loses against refitting every day, on the first ``regularity.outcome_days`` test
days of every confirmation cohort. Women minus men by least squares adjusted
for time in range, CV and mean glucose at startup and the cohort (HC3); Holm
within each outcome and horizon. Patients of another gender or without the
datum are left out of the comparison and counted. The study was not designed for
this comparison; it reports differences with their intervals.

Inputs:
    results/61_regularity_analysis/patient_outcomes.csv
    <derived_data>/runs/06_patient_characteristics/patients.parquet   (sex per patient)
    <derived_data>/runs/60_regularity_indices/indices.parquet         (time in range, CV and mean glucose)
    <derived_data>/runs/50_confirmation_runs/day_metrics.parquet      (RMSE and Clarke zones by sex, R6)

Outputs (results/62_sex_differences/):
    differences.csv      per outcome, model and horizon (0: mean): women, men, raw means, women minus men
                         (points), 95 % CI, p, Holm p
    counts.csv           per cohort: women, men, other genders and not recorded among the patients with outcomes
    metrics_by_sex.csv   per cohort, model, arm, horizon and sex group: n, RMSE, MAE, MARD, bias, Clarke zones (R6)
    differences.png
    summary.md, run_manifest.json

Usage:
    python scripts/62_sex_differences.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain.envguard import ensure_project_env  # noqa: E402  (standard library only)

ensure_project_env(__file__)  # re-runs this script inside the project environment if needed

from t1dretrain.codecheck import verify_code  # noqa: E402  (standard library only)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from t1dretrain.compute import detect_resources  # noqa: E402
from t1dretrain.config import CONFIG_DIR, RESULTS_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.eval import sex_differences as sd  # noqa: E402
from t1dretrain.eval.regularity_analysis import OUTCOMES  # noqa: E402
from t1dretrain.eval.stage0_report import SUMS, color, finish, label  # noqa: E402
from t1dretrain.features.regularity import ADJUSTERS  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, set_style  # noqa: E402
from t1dretrain.sim.stage0 import is_twin  # noqa: E402

SCRIPT_ID = "62_sex_differences"
SOURCES = {"patients": ("06_patient_characteristics", "patients.parquet"),
           "indices": ("60_regularity_indices", "indices.parquet"),
           "days": ("50_confirmation_runs", "day_metrics.parquet")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    derived = derived_data_dir(args.paths)
    outcomes_file = RESULTS_DIR / "61_regularity_analysis" / "patient_outcomes.csv"
    paths = {name: derived / "runs" / script / file for name, (script, file) in SOURCES.items()}
    for path in [outcomes_file, *paths.values()]:
        if not path.exists():
            raise SystemExit(f"{path} not found: run the script that writes it first (06, 50, 60 or 61)")
    settings = {"min_per_sex": int(config.get("sex_differences", {}).get("min_per_sex", 15)),
                "outcome_days": int(config["regularity"]["outcome_days"])}
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()

    outcomes = pd.read_csv(outcomes_file, dtype={"patient_id": "string"})
    patients = pd.read_parquet(paths["patients"])
    adjust = [a for a in ADJUSTERS]
    indices = pd.read_parquet(paths["indices"])
    adjusters = indices[["cohort", "patient_id"] + [a for a in adjust if a in indices]]
    table = sd.sex_differences(outcomes, patients, adjusters, adjust, settings["min_per_sex"])
    counts = count_by_cohort(outcomes, patients)
    days = pd.read_parquet(paths["days"])
    metrics = metrics_by_sex(days, patients, settings["outcome_days"])
    print(f"{len(table)} comparisons; {int(counts['women'].sum())} women and {int(counts['men'].sum())} men")

    out.csv(table.round(5), "differences")
    out.csv(counts, "counts")
    out.csv(metrics, "metrics_by_sex")
    if not table.empty and "difference" in table:
        out.figure(plot_differences(table), "differences")
    out.text(write_summary(table, counts, settings, release, resources), "summary.md")
    out.manifest(settings=settings, notes="Women minus men in the outcomes of script 61 (SAGER, D109).")
    print(f"Done. Results in {out.dir}")


def count_by_cohort(outcomes: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Per cohort, the patients with outcomes by sex group."""
    people = outcomes[["cohort", "patient_id"]].drop_duplicates()
    people = people.assign(dataset=people["cohort"].str.split("/").str[0])
    people = people.merge(sd.sex_groups(patients), on=["dataset", "patient_id"], how="left")
    people["sex_group"] = people["sex_group"].fillna("not recorded")
    table = people.pivot_table(index="cohort", columns="sex_group", values="patient_id", aggfunc="count", fill_value=0)
    for column in ("women", "men", "other", "not recorded"):
        if column not in table:
            table[column] = 0
    return table[["women", "men", "other", "not recorded"]].reset_index()


def metrics_by_sex(days: pd.DataFrame, patients: pd.DataFrame, outcome_days: int) -> pd.DataFrame:
    """R6: n, RMSE, MAE, MARD, bias and Clarke zones per cohort, model, arm, horizon and sex group."""
    d = days[days["days_since_anchor"] < outcome_days]
    d = d.assign(dataset=d["cohort"].str.split("/").str[0]).merge(sd.sex_groups(patients),
                                                                   on=["dataset", "patient_id"], how="left")
    d["sex_group"] = d["sex_group"].fillna("not recorded")
    if d.empty:
        return pd.DataFrame()
    return finish(d.groupby(["cohort", "model", "policy", "horizon_min", "sex_group"], as_index=False)[SUMS].sum()
                  ).round(4)


def plot_differences(table: pd.DataFrame) -> plt.Figure:
    """Women minus men, mean over the horizons, for every full model (and what its inputs add), per outcome."""
    mean = table[(table["horizon_min"] == 0) & table["difference"].notna()]
    models = [m for m in dict.fromkeys(mean["model"]) if not is_twin(m)]
    fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(3.2 * len(OUTCOMES), 0.42 * len(models) + 1.6),
                             sharey=True)
    for ax, outcome in zip(np.atleast_1d(axes), OUTCOMES):
        part = mean[mean["outcome"] == outcome].set_index("model")
        for y, model in enumerate(models):
            if model not in part.index:
                continue
            r = part.loc[model]
            base = model.split(":")[0]
            ax.errorbar(r["difference"], y, xerr=[[r["difference"] - r["low"]], [r["high"] - r["difference"]]],
                        fmt="o", color=color(base), mfc="white" if ":" in model else color(base), ms=5, capsize=0)
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_title(outcome.replace("_", " "), fontsize=9)
        ax.set_xlabel("Women minus men (points)")
    first = np.atleast_1d(axes)[0]
    first.set_yticks(range(len(models)))
    first.set_yticklabels([label(m.split(":")[0]) + (": what inputs add" if ":" in m else "") for m in models])
    first.invert_yaxis()
    fig.suptitle("Women minus men, adjusted for glycaemic control and cohort (mean over horizons, 95 % CI)",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def write_summary(table: pd.DataFrame, counts: pd.DataFrame, settings: dict, release, resources) -> str:
    lines = [f"# {SCRIPT_ID} - Women and men", "", f"Code release {release}.", resources.describe(), "",
             "Outcomes of script 61 (first {} test days), women minus men by least squares adjusted for time in "
             "range, CV and mean glucose at startup and the cohort (HC3); Holm within each outcome and horizon "
             "across models. A comparison needs at least {} women and {} men. Not a test of a hypothesis: the "
             "study was not designed for it.".format(settings["outcome_days"], settings["min_per_sex"],
                                                     settings["min_per_sex"]), "",
             "## Patients with outcomes, per cohort", "", markdown_table(counts)]
    if not table.empty and "p_holm" in table:
        mean = table[table["horizon_min"] == 0]
        hits = table[table["p_holm"] < 0.05]
        lines += ["", "## Overview", "",
                  f"- comparisons: {int(table['difference'].notna().sum())}; Holm p < 0.05: {len(hits)}",
                  f"- largest absolute difference (mean over horizons): "
                  f"{mean['difference'].abs().max():.2f} points" if mean["difference"].notna().any() else "",
                  "", "## Mean over the horizons", "",
                  markdown_table(mean[["outcome", "model", "women", "men", "mean_women", "mean_men", "difference",
                                       "low", "high", "p", "p_holm"]].round(3))]
        if len(hits):
            lines += ["", "## Holm p < 0.05 (any horizon)", "",
                      markdown_table(hits[["outcome", "model", "horizon_min", "women", "men", "difference", "low",
                                           "high", "p_holm"]].round(3))]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
