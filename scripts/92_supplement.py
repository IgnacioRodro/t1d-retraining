"""92 - The tables and the figure of the supplementary material, from the tables of the earlier scripts (D111).

Like scripts 90 and 91, nothing is recomputed: this script selects, orients and
formats (``eval.supplement_tables``). The supplement is contained on purpose
(D105): it holds what the paper points to and what a reviewer is likely to ask
for, and lists the rest of the results as available.

Tables (each as the body of a LaTeX table; captions stay in the supplement, which wraps it with ``\\input``):
    tableS1_regularity   primary regularity indices against the four per-patient outcomes, per full model (61)
    tableS2_sex          patients with outcomes by sex or gender, per confirmation cohort (62)
    tableS3_error        RMSE (mg/dL) frozen and refitted and Clarke zones of the refitted model, per cohort (51)
    tableS4_lows         sensitivity and precision of the forecast lows below 70 mg/dL, per cohort (51)
Figure:
    figS1_sex            women minus men in the four outcomes (mean over horizons), per model (62)

Inputs: results/<script>/ of scripts 51 (replication, metrics_by_arm, low_glucose), 61 (associations) and 62
(counts, differences). Settings: ``supplement`` of the config.

Outputs (results/92_supplement/):
    tableS<n>_<name>.csv/.tex, figS1_sex.pdf/.png (final size) and figS1_sex_data.csv
    summary.md, run_manifest.json

Usage:
    python scripts/92_supplement.py
    python scripts/92_supplement.py --results /path/to/results   # read the tables of another results folder
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
from t1dretrain.config import RESULTS_DIR, load_project_config  # noqa: E402
from t1dretrain.eval import paper_tables as pt  # noqa: E402
from t1dretrain.eval import supplement_tables as st  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import (PAPER_COHORTS, PAPER_FAMILIES, TEXT_SECONDARY, paper_figure,  # noqa: E402
                                 panel_letter, set_paper_style)
from t1dretrain.sim.stage0 import split_name  # noqa: E402

SCRIPT_ID = "92_supplement"
OUTCOME_HEADERS = {"skill_never": "Frozen skill", "ageing": "Aging", "gain_refit": "Gain from refitting",
                   "loss_daily": "Loss to daily"}
OUTCOME_TITLES = {"skill_never": "Skill of the frozen model", "ageing": "Aging (per week)",
                  "gain_refit": "Gain from refitting", "loss_daily": "Loss to daily refitting"}


class Inputs:
    """Reads the tables of the earlier scripts and remembers which ones were used."""

    def __init__(self, results: Path) -> None:
        self.results = results
        self.used: list[str] = []

    def csv(self, script: str, name: str) -> pd.DataFrame:
        target = self.results / script / f"{name}.csv"
        if not target.exists():
            raise SystemExit(f"{target} not found: run scripts/{script}.py first (or pass --results)")
        self.used.append(f"{script}/{name}.csv")
        return pd.read_csv(target, low_memory=False)


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
    cfg = config["supplement"]
    inputs = Inputs(args.results.resolve())
    out = ScriptOutput(SCRIPT_ID, resources=resources)

    replication = inputs.csv("51_confirmation_analysis", "replication")
    reg = cfg["regularity"]
    every = list(dict.fromkeys(cfg["models"] + cfg["lows_models"] + reg["models"]))
    labels = {m: PAPER_FAMILIES.get(split_name(m)[0], (m,))[0] for m in every}

    s1 = st.regularity_table(inputs.csv("61_regularity_analysis", "associations"), reg["models"], reg["outcomes"],
                             list(reg["indices"]))
    s2 = st.sex_counts_table(inputs.csv("62_sex_differences", "counts"), PAPER_COHORTS,
                             config["paper_tables"]["replication_cohorts"])
    s3 = st.error_table(inputs.csv("51_confirmation_analysis", "metrics_by_arm"), replication, cfg["cohorts"],
                        cfg["models"])
    s4 = st.lows_table(inputs.csv("51_confirmation_analysis", "low_glucose"), replication, cfg["cohorts"],
                       cfg["lows_models"])
    tables = {
        "tableS1_regularity": (s1, st.regularity_cells(s1, reg["outcomes"], reg["indices"], labels),
                               [("index", "Index", "X"), ("model", "Model", "l")]
                               + [(o, OUTCOME_HEADERS.get(o, o), "c") for o in reg["outcomes"]]),
        "tableS2_sex": (s2, st.sex_counts_cells(s2),
                        [("label", "Cohort", "X"), ("women", "Women", "r"), ("men", "Men", "r"),
                         ("other", "Other gender", "r"), ("not recorded", "Not recorded", "r")]),
        "tableS3_error": (s3, st.error_cells(s3, PAPER_COHORTS, labels),
                          [("cohort", "Cohort", "X"), ("model", "Model", "l"), ("rmse_frozen", "RMSE, frozen", "c"),
                           ("rmse_refit", "RMSE, refitted", "c"), ("ab", r"Zones A+B (\%)", "c"),
                           ("d", r"Zone D (\%)", "c")]),
        "tableS4_lows": (s4, st.lows_cells(s4, PAPER_COHORTS, labels),
                         [("cohort", "Cohort", "X"), ("model", "Model", "l"), ("state", "State", "l"),
                          ("sensitivity_30", "Sens., 30", "r"), ("precision_30", "Prec., 30", "r"),
                          ("sensitivity_60", "Sens., 60", "r"), ("precision_60", "Prec., 60", "r")]),
    }
    for name, (frame, cells, columns) in tables.items():
        out.csv(frame, name)
        out.text(pt.latex_body(cells, columns), f"{name}.tex")
        print(f"{name}: {len(frame)} rows")

    differences = inputs.csv("62_sex_differences", "differences")
    set_paper_style()
    fig, data = fig_sex(differences, reg["models"], reg["outcomes"], labels)
    out.figure(fig, "figS1_sex", dpi=300, formats=("pdf", "png"), exact_size=True)
    out.csv(data, "figS1_sex_data")

    notes = checks(tables)
    out.text(write_summary(tables, s1, differences, notes, args.results, release, resources), "summary.md")
    out.manifest(settings={"results": str(args.results), "inputs": sorted(set(inputs.used)), **cfg},
                 notes="Tables and figure of the supplementary material from the tables of the earlier scripts.")
    print(f"Done. Results in {out.dir}")


def fig_sex(differences: pd.DataFrame, models: list[str], outcomes: list[str],
            labels: dict[str, str]) -> tuple[plt.Figure, pd.DataFrame]:
    """Women minus men (mean over the horizons) with 95 % intervals: one panel per outcome, one line per full model
    (filled) and per what its insulin and carbohydrate inputs add (open)."""
    mean = differences[(differences["horizon_min"] == 0) & differences["difference"].notna()]
    rows = [m for m in models if m in set(mean["model"])]
    rows += [f"{m}:inputs" for m in models if f"{m}:inputs" in set(mean["model"])]
    fig = paper_figure(14 + 4.2 * len(rows))
    axes = fig.subplots(1, len(outcomes), sharey=True)
    shown = []
    for k, (ax, outcome) in enumerate(zip(np.atleast_1d(axes), outcomes)):
        part = mean[mean["outcome"] == outcome].set_index("model")
        for y, model in enumerate(rows):
            if model not in part.index:
                continue
            r = part.loc[model]
            base = model.split(":")[0]
            colour = PAPER_FAMILIES.get(split_name(base)[0], ("", TEXT_SECONDARY))[1]
            marker = PAPER_FAMILIES.get(split_name(base)[0], ("", "", "o"))[2]
            ax.plot([r["low"], r["high"]], [y, y], color=colour, linewidth=0.8)
            ax.plot(r["difference"], y, marker=marker, color=colour, markersize=3.6, linestyle="none",
                    markerfacecolor="white" if ":" in model else colour)
            shown.append({"outcome": outcome, "model": model, "difference": r["difference"], "low": r["low"],
                          "high": r["high"], "p_holm": r.get("p_holm", np.nan)})
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.6, zorder=1)
        ax.set_title(OUTCOME_TITLES.get(outcome, outcome), loc="center")
        ax.set_xlabel("Women minus men (points)")
        ax.grid(axis="y", visible=False)
        panel_letter(ax, "abcdefgh"[k])
    first = np.atleast_1d(axes)[0]
    first.set_yticks(range(len(rows)))
    first.set_yticklabels([labels.get(m.split(":")[0], m) + (": inputs" if ":" in m else "") for m in rows])
    first.set_ylim(len(rows) - 0.5, -0.5)
    return fig, pd.DataFrame(shown)


def checks(tables: dict) -> list[str]:
    """What the summary should flag: tables that came out empty and cells that do not apply."""
    notes = []
    for name, (frame, cells, _) in tables.items():
        if frame.empty:
            notes.append(f"{name}: empty")
        missing = int((cells == pt.NOT_APPLICABLE).to_numpy().sum())
        if missing:
            notes.append(f"{name}: {missing} cells without a value ('--')")
    return notes


def write_summary(tables: dict, s1: pd.DataFrame, differences: pd.DataFrame, notes: list[str], results: Path,
                  release, resources) -> str:
    lines = [f"# {SCRIPT_ID} - Supplementary material", "", f"Code release {release}. Read from `{results}`.",
             resources.describe(), "", "## Warnings", ""]
    lines += [f"- {n}" for n in notes] or ["- none"]
    if not s1.empty:
        lines += ["", "## Overview", "",
                  f"- S1: {len(s1)} associations; smallest Holm p {s1['p_holm'].min():.3f}; largest absolute "
                  f"coefficient {s1['coef'].abs().max():.2f} points per SD"]
    mean = differences[differences["horizon_min"] == 0]
    if "difference" in mean and mean["difference"].notna().any():
        lines.append(f"- Figure S1: women minus men within {mean['difference'].min():.2f} to "
                     f"{mean['difference'].max():.2f} points (mean over horizons); Holm p < 0.05 at any horizon: "
                     f"{int((differences['p_holm'] < 0.05).sum())}")
    for name, (_, cells, columns) in tables.items():
        frame = pd.DataFrame({h: cells[c].astype(str).str.replace(r"\textminus{}", "-", regex=False)
                              for c, h, _ in columns})
        lines += ["", f"## {name}", "", markdown_table(frame)]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
