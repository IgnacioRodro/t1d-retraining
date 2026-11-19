"""91 - The tables of the paper, written as LaTeX from the tables of the earlier scripts (HANDOFF 2, D105).

Like script 90 for the figures, nothing is recomputed here: every number comes
from a file that an earlier script wrote; this script selects, turns skill into
percentages and formats (``eval.paper_tables``). The one derived number is the
share of the daily refit's gain that the selected interval keeps, the ratio of
two gains that script 51 published. Text that no result file holds (therapy,
sensor, inputs and role of each cohort; class, inputs, refits tested and stages
of each model; the recommended use of each model) is in ``paper_tables`` of the
config.

Tables (each as the body of a LaTeX table; the caption and the notes stay in the
manuscript, which wraps it with ``\\input``):
    table1_cohorts       datasets and cohorts: eligible and analysed patients, median usable days, age, share of
                         women (04, 06, 51, 73, 74)
    table2_models        per model: class, inputs, starting point as the pool chose it (size of the population model,
                         what fine-tuning trains), refits tested in stage 3 and stages (11; text in the config)
    table3_guide         per model: interval and mechanism of the confirmation, skill frozen and refitted in the
                         screening, share of the daily gain kept, compute per test day, recommended use
                         (11, 43, 51, 74)
    table4_replication   per confirmation cohort: the findings of the screening that replicated (51)

Inputs: results/<script>/ of scripts 04 (cohort_summary, cohorts), 11 (chosen_settings.yaml), 43 (cost),
51 (replication, replication_summary), 74 (skill) and, if present, 06 (characteristics) and 73 (per_patient).

Outputs (results/91_paper_tables/):
    table1_cohorts.csv/.tex, table2_models.csv/.tex, table3_guide.csv/.tex, table4_replication.csv/.tex
    tables.png           a preview of the four tables
    summary.md, run_manifest.json

Usage:
    python scripts/91_paper_tables.py
    python scripts/91_paper_tables.py --results /path/to/results   # read the tables of another results folder
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain.envguard import ensure_project_env  # noqa: E402  (standard library only)

ensure_project_env(__file__)  # re-runs this script inside the project environment if needed

from t1dretrain.codecheck import verify_code  # noqa: E402  (standard library only)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from t1dretrain.compute import detect_resources  # noqa: E402
from t1dretrain.config import RESULTS_DIR, load_project_config  # noqa: E402
from t1dretrain.eval import paper_tables as pt  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import PAPER_COHORTS, PAPER_FAMILIES  # noqa: E402
from t1dretrain.sim.stage0 import split_name  # noqa: E402

SCRIPT_ID = "91_paper_tables"
SCREEN = "screen (stage 3)"

TABLE1_COLUMNS = [("dataset", "Dataset", "l"), ("cohort", "Cohort", "l"), ("therapy_sensor", "Therapy; sensor", "X"),
                  ("inputs", "Inputs", "l"), ("eligible", "Eligible", "r"), ("analysed", "Analyzed", "r"),
                  ("median_days", "Days", "r"), ("age", "Age (y)", "l"), ("female_pct", r"Women (\%)", "r"),
                  ("role", "Role in the study", "X")]
TABLE2_COLUMNS = [("label", "Model", "l"), ("kind", "Class", "X"), ("inputs", "Inputs", "l"),
                  ("start", "Starting point", "X"), ("refits", "Refits tested", "X"), ("stages", "Stages", "l")]
TABLE3_COLUMNS = [("label", "Model", "l"), ("interval", "Interval", "l"), ("mechanism", "Mechanism", "X"),
                  ("frozen", r"Skill frozen (\%)", "c"), ("refitted", r"Skill refitted (\%)", "c"),
                  ("kept", r"Kept (\%)", "c"), ("seconds", "Cost (s/day)", "c"), ("use", "Recommended use", "X")]
TABLE4_COLUMNS = [("label", "Cohort", "X"), ("patients", "Patients", "r"), ("gain", "Gain", "c"),
                  ("full", "Full", "c"), ("daily", "Daily", "c"), ("mechanism", "Mechanism", "c")]


class Inputs:
    """Reads the tables of the earlier scripts and remembers which ones were used."""

    def __init__(self, results: Path) -> None:
        self.results = results
        self.used: list[str] = []

    def path(self, script: str, name: str) -> Path:
        target = self.results / script / name
        if not target.exists():
            raise SystemExit(f"{target} not found: run scripts/{script}.py first (or pass --results)")
        self.used.append(f"{script}/{name}")
        return target

    def csv(self, script: str, name: str) -> pd.DataFrame:
        return pd.read_csv(self.path(script, f"{name}.csv"), low_memory=False)

    def optional_csv(self, script: str, name: str) -> pd.DataFrame | None:
        return self.csv(script, name) if (self.results / script / f"{name}.csv").exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=RESULTS_DIR,
                        help="folder with the results of the earlier scripts (default: results/)")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    cfg = config["paper_tables"]
    inputs = Inputs(args.results.resolve())
    out = ScriptOutput(SCRIPT_ID, resources=resources)

    replication = inputs.csv("51_confirmation_analysis", "replication")
    summary = inputs.csv("51_confirmation_analysis", "replication_summary")
    skill = inputs.csv("74_family_comparison", "skill")
    labels = {model: PAPER_FAMILIES.get(split_name(model)[0], (model,))[0] for model in cfg["guide_models"]}

    # Table 1: datasets and cohorts.
    analysed = {key: int(n) for key, n in replication.groupby("cohort")["patients"].max().items()}
    analysed["replace_bg/confirm"] = analysed.pop("replace_bg/core", None)
    screen = skill[skill["source"] == SCREEN]["patients"]
    analysed["replace_bg/screen"] = int(screen.max()) if len(screen) else None
    wearables = inputs.optional_csv("73_wearables", "per_patient")
    if wearables is not None:
        for key, n in wearables.groupby("cohort")["patient_id"].nunique().items():
            analysed.setdefault(key, int(n))
    table1 = pt.cohort_table(cfg["cohorts"], inputs.csv("04_define_splits", "cohort_summary"),
                             inputs.csv("04_define_splits", "cohorts"),
                             inputs.optional_csv("06_patient_characteristics", "characteristics"), analysed)

    # Table 2: models, with the starting points the pool chose.
    chosen = yaml.safe_load(inputs.path("11_pretrain_population", "chosen_settings.yaml").read_text()) or {}
    table2 = pt.models_table(cfg["models"], chosen)

    # Table 3: practical guide per model.
    finetune = {model: (chosen.get(split_name(model)[0]) or {}).get("finetune", {}).get("train")
                for model in cfg["guide_models"]}
    table3 = pt.guide_table(skill, replication, inputs.csv("43_stage3_context", "cost"), cfg["guide_models"],
                            finetune, cfg.get("uses", {}), labels)

    # Table 4: replication per confirmation cohort.
    table4 = pt.replication_table(summary, replication, cfg["replication_cohorts"], PAPER_COHORTS)

    tables = {"table1_cohorts": (table1, pt.cohort_cells(table1), TABLE1_COLUMNS),
              "table2_models": (table2, pt.models_cells(table2), TABLE2_COLUMNS),
              "table3_guide": (table3, pt.guide_cells(table3), TABLE3_COLUMNS),
              "table4_replication": (table4, pt.replication_cells(table4), TABLE4_COLUMNS)}
    for name, (frame, cells, columns) in tables.items():
        out.csv(frame, name)
        out.text(pt.latex_body(cells, columns), f"{name}.tex")
        print(f"{name}: {len(frame)} rows")
    out.figure(preview(tables), "tables.png", dpi=150)
    notes = checks(table1, table2, table3, table4)
    out.text(write_summary(tables, notes, args.results, release, resources), "summary.md")
    out.manifest(settings={"results": str(args.results), "inputs": sorted(set(inputs.used))},
                 notes="Paper tables written as LaTeX from the tables of the earlier scripts; nothing is recomputed.")
    print(f"Done. Results in {out.dir}")


def checks(table1: pd.DataFrame, table2: pd.DataFrame, table3: pd.DataFrame, table4: pd.DataFrame) -> list[str]:
    """What the summary should flag: values still missing."""
    notes = []
    pending = table2[table2["start"] == pt.PENDING]["family"].tolist()
    if pending:
        notes.append(f"table 2: no population model in 11's chosen_settings.yaml for {pending}")
    pending = table1[table1["female_status"] == "pending"]["key"].tolist()
    if pending:
        notes.append(f"table 1: share of women pending for {pending} (run scripts/06_patient_characteristics.py)")
    none = table1[table1["female_status"] == "not recorded"]["key"].tolist()
    if none:
        notes.append(f"table 1: no sex or gender recorded for {none} (n.r.)")
    none = table1[table1["age_status"] == "not recorded"]["key"].tolist()
    if none:
        notes.append(f"table 1: no age recorded for {none} (n.r.)")
    for column in ("kept_low", "seconds_per_day"):
        gaps = table3[table3[column].isna()]["model"].tolist()
        if gaps:
            notes.append(f"table 3: no {column} for {gaps}")
    if table4.empty:
        notes.append("table 4: no cohort of paper_tables.replication_cohorts in replication_summary.csv")
    return notes


def plain(text: str) -> str:
    """A LaTeX cell as plain text for the preview."""
    text = str(text).replace(r"\textsuperscript{a}", " (a)").replace(pt.PENDING, "…").replace("$\\lambda=1$", "λ=1")
    return re.sub(r"\\([%&_$#])", r"\1", text).replace("--", "–")


def preview(tables: dict) -> plt.Figure:
    """The tables as plain text, one under the other."""
    heights = [len(cells) + 2 for _, cells, _ in tables.values()]
    fig, axes = plt.subplots(len(tables), 1, figsize=(16, 0.32 * sum(heights) + 1),
                             gridspec_kw={"height_ratios": heights})
    for ax, (name, (_, cells, columns)) in zip(axes, tables.items()):
        ax.axis("off")
        ax.set_title(name, loc="left", fontsize=10)
        body = [[plain(row[c]) for c, _, _ in columns] for _, row in cells.iterrows()]
        table = ax.table(cellText=body, colLabels=[plain(h) for _, h, _ in columns], loc="upper left",
                         cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width(list(range(len(columns))))
    fig.tight_layout()
    return fig


def write_summary(tables: dict, notes: list[str], results: Path, release, resources) -> str:
    lines = [f"# {SCRIPT_ID} - The tables of the paper", "",
             f"Code release {release}. Read from `{results}`.", resources.describe(), "",
             "Each `<table>.tex` is the body of a LaTeX table (booktabs, tabularx); the manuscript wraps it with "
             "its caption. `\\pend` marks a value still to add (macro of the manuscript); `--`, a cell that does "
             "not apply; `n.r.`, a dataset that does not record it.", "", "## Warnings", ""]
    lines += [f"- {n}" for n in notes] or ["- none"]
    for name, (_, cells, columns) in tables.items():
        frame = pd.DataFrame({plain(h): cells[c].map(plain) for c, h, _ in columns})
        lines += ["", f"## {name}", "", markdown_table(frame)]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
