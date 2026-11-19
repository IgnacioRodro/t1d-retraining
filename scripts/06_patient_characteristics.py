"""06 - Sex or gender and age of the patients of every cohort, for table 1 of the paper (HANDOFF 2, D108; SAGER).

The journal asks to report sex and gender (SAGER guidelines). Each dataset
records it its own way (``data.demographics``): REPLACE-BG and DCLP3 a Gender
column (M, F), Loop a baseline survey question on gender (1 male, 2 female, 3
non-binary), BrisT1D a free-text answer on gender; T1D-UOM and HUPA-UCM publish
none. This script reads the static patients table of every dataset with its
loader (only the ``patients`` task, in seconds; the harmonised data of script
02 are not touched), normalises the answer to female, male or other, and
counts per cohort of script 04 and per group of REPLACE-BG: patients, women,
men, other genders, not recorded, the share of women among those recorded and
the age. Script 91 takes the share of women into table 1.

Inputs: config/paths.yaml (raw folders); results/04_define_splits/cohorts.csv (patients of every cohort).

Outputs (results/06_patient_characteristics/):
    characteristics.csv   per cohort (``<dataset>/<cohort>``) and REPLACE-BG group (``replace_bg/<group>``):
                          patients, women, men, other_gender, not_recorded, women_pct, age (median, quartiles)
    recorded.csv          per dataset: patients in the loader's table and with a recorded sex or gender
    checks_loop_survey_match.csv   Loop: roster participants with a baseline survey and age agreement
    characteristics.png   share of women, men, other genders and not recorded per cohort
    summary.md, run_manifest.json
    <derived_data>/runs/06_patient_characteristics/patients.parquet   per patient: dataset, patient_id, sex
                          (female, male, other or empty) and age, for the comparison by sex of script 62

Usage:
    python scripts/06_patient_characteristics.py
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
import pandas as pd  # noqa: E402

from t1dretrain.compute import detect_resources  # noqa: E402
from t1dretrain.config import CONFIG_DIR, RESULTS_DIR, derived_data_dir, load_paths, load_project_config  # noqa: E402
from t1dretrain.data.demographics import cohort_characteristics, normalize_sex, patients_table  # noqa: E402
from t1dretrain.io import LOADERS  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import DATASET_LABELS, TEXT_SECONDARY, set_style  # noqa: E402

SCRIPT_ID = "06_patient_characteristics"
SHARES = (("women", "#c4458a", "Women"), ("men", "#2a78d6", "Men"), ("other_gender", "#1baf7a", "Other gender"),
          ("not_recorded", "#d9d8d4", "Not recorded"))


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
    cohorts_file = RESULTS_DIR / "04_define_splits" / "cohorts.csv"
    if not cohorts_file.exists():
        raise SystemExit(f"{cohorts_file} not found: run scripts/04_define_splits.py first")
    cohorts = pd.read_csv(cohorts_file, dtype={"patient_id": "string", "group": "string"})
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()

    paths = load_paths(args.paths)
    tables, recorded, checks, notes = [], [], [], []
    for dataset in sorted(set(cohorts["dataset"])):
        root = paths.get(dataset)
        if dataset not in LOADERS or root is None or not Path(root).exists():
            notes.append(f"{dataset}: no loader or raw folder ({root}); left out (script 91 marks it pending)")
            continue
        table, report = patients_table(dataset, Path(root))
        sex_raw = table["sex"] if "sex" in table else pd.Series(pd.NA, index=table.index, dtype="string")
        sex = normalize_sex(sex_raw)
        unknown = sorted({str(v) for v, s in zip(sex_raw, sex) if pd.notna(v) and str(v).strip() and pd.isna(s)})
        if unknown:
            notes.append(f"{dataset}: answers not mapped to female, male or other (counted as not recorded): "
                         f"{unknown[:10]}")
        tables.append(pd.DataFrame({"dataset": dataset, "patient_id": table["patient_id"].astype("string"),
                                    "sex": sex, "age": pd.to_numeric(table["age"], errors="coerce") if "age" in table
                                    else float("nan")}))
        recorded.append({"dataset": dataset, "patients": int(len(table)), "sex_recorded": int(sex.notna().sum())})
        checks += [row for row in report.checks if row.get("check") == "loop_survey_match"]
        print(f"{dataset}: {len(table)} patients, sex or gender recorded for {int(sex.notna().sum())}")

    patients = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
        columns=["dataset", "patient_id", "sex", "age"])
    runs = derived_data_dir(args.paths) / "runs" / SCRIPT_ID
    runs.mkdir(parents=True, exist_ok=True)
    patients.to_parquet(runs / "patients.parquet", index=False)
    read = {row["dataset"] for row in recorded}
    characteristics = cohort_characteristics(cohorts[cohorts["dataset"].isin(read)], patients)
    out.csv(characteristics.round(2), "characteristics")
    out.csv(pd.DataFrame(recorded), "recorded")
    if checks:
        out.csv(pd.DataFrame(checks), "checks_loop_survey_match")
    out.figure(plot_shares(characteristics), "characteristics.png")
    out.text(write_summary(characteristics, pd.DataFrame(recorded), pd.DataFrame(checks), notes, release,
                           resources), "summary.md")
    out.manifest(settings={"cohorts": str(cohorts_file)},
                 notes="Sex or gender and age per cohort, from the loaders' patients tables (SAGER, D108).")
    print(f"Done. Results in {out.dir}")


def plot_shares(characteristics: pd.DataFrame) -> plt.Figure:
    """One bar per cohort and group: share of women, men, other genders and not recorded."""
    frame = characteristics.set_index("key")
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(frame) + 1.2))
    left = pd.Series(0.0, index=frame.index)
    for column, colour, text in SHARES:
        share = 100 * frame[column] / frame["patients"].where(frame["patients"] > 0)
        ax.barh(range(len(frame)), share.fillna(0), left=left, color=colour, label=text, height=0.7)
        left += share.fillna(0)
    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels([f"{DATASET_LABELS.get(k.split('/')[0], k.split('/')[0])} {k.split('/', 1)[1]} "
                        f"({int(n)})" for k, n in frame["patients"].items()])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Patients of the cohort (%)")
    ax.legend(ncols=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.tick_params(axis="y", colors=TEXT_SECONDARY)
    fig.suptitle("Sex or gender as each dataset records it", fontweight="bold", x=0.01, ha="left")
    return fig


def write_summary(characteristics: pd.DataFrame, recorded: pd.DataFrame, checks: pd.DataFrame, notes: list[str],
                  release, resources) -> str:
    lines = [f"# {SCRIPT_ID} - Sex or gender and age per cohort", "", f"Code release {release}.",
             resources.describe(), "",
             "Sex or gender as each dataset records it, normalised to female, male or other (`data.demographics`, "
             "D108); the share of women is over the patients with a recorded answer. T1D-UOM and HUPA-UCM publish "
             "none.", "", "## Warnings", ""]
    lines += [f"- {n}" for n in notes] or ["- none"]
    lines += ["", "## Per dataset", "", markdown_table(recorded) if not recorded.empty else "(none)",
              "", "## Per cohort and REPLACE-BG group", "", markdown_table(characteristics.round(1))]
    if not checks.empty:
        lines += ["", "## Loop: link of the baseline survey to the roster", "",
                  "Participants of the roster with a baseline survey, and how often the survey's age is within a "
                  "year of the roster's (a check that SubjectID is the roster's PtID).", "",
                  markdown_table(checks.drop(columns=["check"], errors="ignore"))]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
