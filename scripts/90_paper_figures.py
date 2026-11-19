"""90 - The figures of the paper, drawn from the tables of the earlier scripts (HANDOFF 2, D102, D104).

Nothing is computed here that changes a result: every number comes from a CSV
that an earlier script wrote (rule R6 keeps the grids and zones there); this
script only selects, converts skill to points and draws. Each figure is the
final printed size of an Elsevier journal (full width 190 mm; at most 240 mm
tall), with 7 pt text (never below 6 pt), panel letters and no title: the
conclusion goes in the caption, in the LaTeX source. One column per horizon
(30, 60 and 120 min). One colour and marker per model family in every figure
(``plotting.PAPER_FAMILIES``). The confirmation is shown apart (D102, option
B): the result figures use the screening, figure 9 the replication, and the
confirmation enters another figure only where the screening has nothing like
it (Loop up to 36 weeks, the closed loop of DCLP3, the forecast lows).

Figures (the list and its order: ``paper_figures.figures``):
    fig01_design               study design and evaluation protocol (cohort sizes from scripts 04 and 50)
    fig02_population_personal  personal only -> population -> fine-tuned -> refitted; LightGBM personal only
    fig03_families             every family against LightGBM, frozen and refitted (74); skill against cost (43)
    fig04_ageing               ageing against a model that cannot age (43); slopes; recency against data (21)
    fig05_policy               window x interval (31); share of the daily gain kept; gain against cost
    fig06_data                 gain week by week, screening and Loop to 36 weeks (31, 51); later anchors (31)
    fig07_mechanism            each mechanism against the restart (41, 51); context (43); IS-LSTM (75)
    fig08_inputs               refitting with and without insulin and carbohydrates (31, 51); wearables (73)
    fig09_replication          the gain at the optimal interval in every cohort next to the screening (31, 51)
    fig10_closed_loop          after the switch to closed loop against its control (72)
    fig11_lows                 shrinkage, alarm curves with the weighted loss, Clarke zones D and E (50, 70, 71)

Each figure is written as a vector PDF and a 300 dpi PNG, with the data it
draws (``<figure>_data.csv``: the source data a journal may ask for). A
figure whose inputs are missing is left out and named in the summary.

Inputs: results/<script>/*.csv of scripts 04, 11, 12, 21, 31, 41, 43, 50, 51, 70-75 (see each figure).

Outputs (results/90_paper_figures/):
    figNN_<name>.pdf, figNN_<name>.png, figNN_<name>_data.csv
    figures.csv          per figure: size, panels, inputs, whether it was drawn
    summary.md, run_manifest.json

Usage:
    python scripts/90_paper_figures.py
    python scripts/90_paper_figures.py --results /path/to/results   # read the tables of another results folder
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain.envguard import ensure_project_env  # noqa: E402  (standard library only)

ensure_project_env(__file__)  # re-runs this script inside the project environment if needed

from t1dretrain.codecheck import verify_code  # noqa: E402  (standard library only)

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.font_manager import FontProperties, findfont  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from t1dretrain.compute import detect_resources  # noqa: E402
from t1dretrain.config import RESULTS_DIR, load_project_config  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import (  # noqa: E402
    MM, PAPER_COHORTS, PAPER_FAMILIES, PAPER_FONTS, PAPER_MAX_HEIGHT_MM, PAPER_MIN_PT, PAPER_SMALL_PT, PAPER_WIDTH_MM,
    TEXT_PRIMARY, TEXT_SECONDARY, panel_letter, paper_figure, set_paper_style,
)
from t1dretrain.sim.stage0 import is_twin, split_name  # noqa: E402

SCRIPT_ID = "90_paper_figures"
HORIZONS = (30, 60, 120)
SCREEN = "screen (stage 3)"
LEARNED = ("lightgbm_ft", "lstm_ft", "tsmixerx_ft", "tide_ft")
COHORT_LABELS = PAPER_COHORTS
ZERO = {"color": TEXT_SECONDARY, "linewidth": 0.6, "zorder": 1}
LEGEND_ROW_MM = 4.0  # height of one row of the legend strip under the panels


class Missing(Exception):
    """An input table of the figure does not exist (the earlier script has not run)."""


@dataclass
class Drawn:
    """A drawn figure: the figure, the data it shows (tidy) and notes for the summary."""

    fig: plt.Figure
    data: pd.DataFrame
    notes: list[str] = field(default_factory=list)


class Inputs:
    """Reads the tables of the earlier scripts and remembers which ones a figure used."""

    def __init__(self, results: Path) -> None:
        self.results = results
        self.used: list[str] = []

    def path(self, script: str, name: str) -> Path:
        target = self.results / script / name
        if not target.exists():
            raise Missing(f"{script}/{name}")
        self.used.append(f"{script}/{name}")
        return target

    def csv(self, script: str, name: str) -> pd.DataFrame:
        return pd.read_csv(self.path(script, f"{name}.csv"), low_memory=False)

    def yaml(self, script: str, name: str) -> dict:
        return yaml.safe_load(self.path(script, name).read_text()) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=RESULTS_DIR,
                        help="folder with the results of the earlier scripts (default: results/)")
    parser.add_argument("--only", nargs="*", default=None, help="draw only these figures (names or numbers)")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


# ============================================================================ main


def main() -> None:
    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    cfg = config.get("paper_figures", {})
    wanted = [str(name) for name in cfg.get("figures", list(FIGURES))]
    if args.only:
        wanted = [name for name in wanted if any(name.startswith(f"fig{int(o):02d}_") if o.isdigit()
                                                 else name == o or name.startswith(f"{o}_") for o in args.only)]
    unknown = [name for name in wanted if name not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figures {unknown}: one of {list(FIGURES)}")
    if not wanted:
        raise SystemExit(f"no figure matches --only {args.only}: use numbers (5), short names (fig05) or full "
                         f"names ({list(FIGURES)[0]})")
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_paper_style()
    font = Path(findfont(FontProperties(family=PAPER_FONTS))).name
    print(f"Font: {font}")
    inputs = Inputs(args.results.resolve())
    settings = {"min_patients": int(cfg.get("min_patients", 20)), "small_cohorts": int(cfg.get("small_cohorts", 10))}

    rows, notes, failed = [], [], []
    for name in wanted:
        builder, width, panels = FIGURES[name]
        inputs.used = []
        row = {"figure": name, "width": width, "width_mm": PAPER_WIDTH_MM[width], "panels": panels}
        try:
            drawn = builder(inputs, settings)
        except Missing as error:
            row.update(drawn=False, height_mm=np.nan, inputs="; ".join(inputs.used), note=f"missing {error}")
            print(f"{name}: left out, missing {error}")
            notes.append(f"{name}: left out, missing {error}")
            rows.append(row)
            continue
        except Exception as error:  # noqa: BLE001  (one figure that fails must not stop the others)
            plt.close("all")
            print(f"{name}: FAILED\n{traceback.format_exc()}")
            row.update(drawn=False, height_mm=np.nan, inputs="; ".join(inputs.used),
                       note=f"error {type(error).__name__}: {error}")
            notes.append(f"{name}: failed with {type(error).__name__}: {error} (the traceback is in the log)")
            failed.append(name)
            rows.append(row)
            continue
        size = drawn.fig.get_size_inches() / MM
        problems = check_text(drawn.fig)
        out.figure(drawn.fig, name, dpi=300, formats=("pdf", "png"), exact_size=True)
        out.csv(drawn.data, f"{name}_data")
        row.update(drawn=True, height_mm=round(float(size[1]), 1), inputs="; ".join(dict.fromkeys(inputs.used)),
                   note=" ".join(drawn.notes + problems))
        rows.append(row)
        notes += [f"{name}: {text}" for text in drawn.notes + problems]
        print(f"{name}: {size[0]:.0f} x {size[1]:.0f} mm" + (f"; {' '.join(problems)}" if problems else ""))

    table = pd.DataFrame(rows)
    out.csv(table, "figures")
    out.text(write_summary(table, notes, font, args.results, release, resources), "summary.md")
    out.manifest(settings={"figures": wanted, "results": str(args.results), "font": font, **settings},
                 notes="Paper figures drawn from the tables of the earlier scripts; nothing is recomputed.")
    print(f"Done. Results in {out.dir}")
    if failed:
        raise SystemExit(f"figures that failed: {failed} (the others were drawn; see the log and summary.md)")


def check_text(fig: plt.Figure) -> list[str]:
    """Problems a printed figure must not have: text outside the page or smaller than 6 pt."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    small = sorted({text.get_text()[:20] for text in fig.findobj(matplotlib.text.Text)
                    if text.get_visible() and text.get_text().strip() and text.get_fontsize() < PAPER_MIN_PT - 1e-6})
    outside = []
    axes = [ax for ax in fig.findobj(matplotlib.axes.Axes) if ax.get_visible()]
    pieces = [(f"panel {k + 1}", ax.get_tightbbox(renderer)) for k, ax in enumerate(axes)]
    pieces += [("legend", legend.get_window_extent(renderer)) for legend in fig.findobj(matplotlib.legend.Legend)]
    pieces += [(text.get_text()[:20], text.get_window_extent(renderer)) for part in [fig, *fig.subfigs]
               for text in part.texts if text.get_text()]
    for name, box in pieces:
        if box is not None and (box.x0 < page.x0 - 1 or box.x1 > page.x1 + 1 or box.y0 < page.y0 - 1
                                or box.y1 > page.y1 + 1):
            outside.append(name)
    problems = []
    if outside:
        problems.append(f"text outside the page: {outside}")
    if small:
        problems.append(f"text below {PAPER_MIN_PT:g} pt: {small}")
    if fig.get_size_inches()[1] / MM > PAPER_MAX_HEIGHT_MM + 0.5:
        problems.append(f"taller than {PAPER_MAX_HEIGHT_MM:g} mm")
    return problems


# ============================================================================ shared pieces


def family(model: str) -> str:
    return split_name(str(model))[0]


def label(model: str) -> str:
    text = PAPER_FAMILIES.get(family(model), (str(model),))[0]
    return f"{text}, glucose only" if is_twin(str(model)) else text


def colour(model: str) -> str:
    return PAPER_FAMILIES.get(family(model), ("", TEXT_SECONDARY))[1]


def marker(model: str) -> str:
    return PAPER_FAMILIES.get(family(model), ("", "", "o"))[2]


def points(frame: pd.DataFrame, columns) -> pd.DataFrame:
    """Skill fractions as points (x 100)."""
    frame = frame.copy()
    for column in columns:
        frame[column] = frame[column] * 100
    return frame


def horizon_grid(fig: plt.Figure, rows: int, heights=None, letters="abcdefgh", sharey_rows=True,
                 legend_rows: int = 1):
    """Rows x 3 panels, one column per horizon, and a strip at the bottom for the legend: (axes, strip). Each row
    is laid out on its own (a subfigure), so long labels in one row do not narrow the panels of another. The top
    row says the horizon; each row starts with its letter."""
    heights = list(heights or [1.0] * rows)
    strip_mm = LEGEND_ROW_MM * legend_rows + 1.5
    unit = sum(heights) / (fig.get_size_inches()[1] / MM - strip_mm)
    parts = np.atleast_1d(fig.subfigures(rows + 1, 1, height_ratios=heights + [strip_mm * unit]))
    axes = np.array([part.subplots(1, len(HORIZONS)) for part in parts[:rows]])
    for column, horizon in enumerate(HORIZONS):
        axes[0, column].set_title(f"{horizon} min", loc="center")
    for row in range(rows):
        panel_letter(axes[row, 0], letters[row])
        if sharey_rows:
            for column in range(1, len(HORIZONS)):
                axes[row, column].sharey(axes[row, 0])
    return axes, parts[rows]


def model_handles(models, filled=True, line=False) -> list[Line2D]:
    return [Line2D([], [], color=colour(m), marker=marker(m), linestyle="-" if line else "none",
                   markerfacecolor=colour(m) if filled else "white", markersize=4, label=label(m)) for m in models]


def state_handles(open_label: str, filled_label: str) -> list[Line2D]:
    grey = TEXT_SECONDARY
    return [Line2D([], [], color=grey, marker="o", linestyle="none", markerfacecolor="white", markersize=4,
                   label=open_label),
            Line2D([], [], color=grey, marker="o", linestyle="none", markerfacecolor=grey, markersize=4,
                   label=filled_label)]


def bottom_legend(strip, handles, ncols: int | None = None) -> None:
    """The legend of the figure, centred in the strip below the panels; fewer columns if a wider font makes it
    wider than the page."""
    ncols = ncols or len(handles)
    while True:
        legend = strip.legend(handles=handles, loc="center", ncols=ncols, frameon=False)
        figure = strip.figure if not hasattr(strip, "get_figure") else strip.get_figure(root=True)
        figure.canvas.draw()
        if ncols == 1 or legend.get_window_extent().width <= figure.bbox.width - 2:
            return
        legend.remove()
        ncols -= 1


def same_limits(axes, axis: str = "y") -> None:
    """One scale for panels drawn each on its own (a row): the union of their limits."""
    get, put = ("get_ylim", "set_ylim") if axis == "y" else ("get_xlim", "set_xlim")
    lows, highs = zip(*(getattr(ax, get)() for ax in axes))
    for ax in axes:
        getattr(ax, put)(min(lows), max(highs))


def versus(pairs: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Every model against ``reference`` from a table of pairs in either order: the other model minus the reference,
    with the interval and the verdict turned round where the reference came first."""
    swap = {"better": "worse", "worse": "better"}
    a = pairs[pairs["against"] == reference].copy()
    b = pairs[pairs["model"] == reference].copy()
    b = b.assign(model=b["against"], against=reference, difference=-b["difference"], low=-b["high"], high=-b["low"],
                 verdict=b["verdict"].replace(swap))
    return pd.concat([a, b], ignore_index=True)


def forest(ax, rows: list[str], frame: pd.DataFrame, models: list[str], row_column: str, value: str = "difference",
           spread: float = 0.22, filled_when=("better", "worse")) -> pd.DataFrame:
    """Dots and 95 % intervals: one line per entry of ``rows`` (top to bottom), one dot per model, offset within the
    line; filled when the verdict is a difference, open for a tie or an uncertain one."""
    offsets = np.linspace(spread, -spread, len(models)) if len(models) > 1 else [0.0]
    shown = []
    for i, row in enumerate(rows):
        for offset, model in zip(offsets, models):
            hit = frame[(frame[row_column] == row) & (frame["model"] == model)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            y = len(rows) - 1 - i + offset
            full = r.get("verdict") in filled_when
            ax.plot([r["low"], r["high"]], [y, y], color=colour(model), linewidth=0.8, zorder=2)
            ax.plot(r[value], y, marker=marker(model), color=colour(model), markersize=3.6, linestyle="none",
                    markerfacecolor=colour(model) if full else "white", zorder=3)
            shown.append(r)
    ax.axvline(0, **ZERO)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([COHORT_LABELS.get(r, r) for r in rows][::-1])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    return pd.DataFrame(shown)


def hide_inner_y(axes) -> None:
    """Shared y labels: only the first column keeps its tick labels."""
    for row in axes:
        for ax in row[1:]:
            ax.tick_params(labelleft=False)


# ============================================================================ figure 1


def fig01_design(inputs: Inputs, settings: dict) -> Drawn:
    """Study design (a) and the protocol of one patient (b)."""
    summary = inputs.csv("04_define_splits", "cohort_summary")
    cohorts = inputs.csv("50_confirmation_runs", "cohorts")
    core = summary[summary["role"] == "core"].iloc[0]
    n_pool, n_screen = int(core["n_pool"]), int(core["n_screen"])
    confirm = cohorts.set_index("cohort")["patients"].astype(int)
    with_anchor = cohorts.set_index("cohort")["with_anchor"].astype(int)

    lines = []
    for cohort in confirm.index:
        if cohort not in COHORT_LABELS:
            continue
        name = "REPLACE-BG hold-out" if cohort.startswith("replace_bg") else COHORT_LABELS[cohort]
        sampled = f" (of {with_anchor[cohort]})" if with_anchor[cohort] > confirm[cohort] + 10 else ""
        lines.append(f"{name}, {confirm[cohort]}{sampled}")
    light, blue, orange = "#f3f2ef", "#e3eefb", "#fbe9e1"
    entries = [("Pool", [f"REPLACE-BG, {n_pool} patients", "", "Population models of", "every family;",
                         "hyper-parameters and", "fine-tuning settings"], light),
               ("Screening", [f"REPLACE-BG, {n_screen} patients", "", "Stage 0: population and personal",
                              "Stage 1: aging of frozen models", "Stage 2: refit policy",
                              "Stage 3: refit mechanism, context", "Rules fixed before the confirmation"], blue),
               ("Confirmation (patients)", lines, orange),
               ("After the confirmation", ["Forecast lows", "Weighted loss", "Recovery after changes", "Wearables",
                                           "Families compared", "Closest precedent", "(IS-LSTM)"], light)]

    fig = paper_figure(88)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.25, 1])
    top, bottom = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    panel_letter(top, "a")
    panel_letter(bottom, "b")
    top.axis("off")
    draw_protocol(bottom)
    flow_boxes(top, entries)  # last: it measures the text in the final layout
    data = pd.DataFrame({"group": ["pool", "screen"] + list(confirm.index),
                         "patients": [n_pool, n_screen] + list(confirm.values),
                         "with_anchor": [np.nan, np.nan] + list(with_anchor[confirm.index].values)})
    return Drawn(fig, data)


def flow_boxes(ax, entries, gap_min: float = 0.035) -> None:
    """Boxes from left to right with arrows between them, each as wide as its text (a title and lines) in the
    font actually used, so that no text leaves its box; the room left over goes to the arrows."""
    fig = ax.figure
    made = []
    for title, lines, _ in entries:
        head = ax.text(0, 1, title, fontweight="bold", va="top", ha="left", transform=ax.transAxes)
        body = ax.text(0, 1, "\n".join(lines), fontsize=PAPER_SMALL_PT, va="top", ha="left", linespacing=1.3,
                       transform=ax.transAxes)
        for text in (head, body):
            text.set_in_layout(False)  # the layout must not depend on where they stand while they are measured
        made.append((head, body))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_axes = ax.transAxes.inverted()

    def size(text):
        box = text.get_window_extent(renderer).transformed(to_axes)
        return box.width, box.height

    padx, pady, between = 0.012, 0.05, 0.03
    sizes = [(size(head), size(body)) for head, body in made]
    widths = [max(h[0], b[0]) + 2 * padx for h, b in sizes]
    height = max(h[1] + between + b[1] for h, b in sizes) + 2 * pady
    gap = max(gap_min, (1.0 - sum(widths)) / max(len(entries) - 1, 1))
    top, x = 0.98, 0.0
    for (title, lines, tone), (head, body), (hsize, _), width in zip(entries, made, sizes, widths):
        ax.add_patch(FancyBboxPatch((x, top - height), width, height, boxstyle="round,pad=0,rounding_size=0.012",
                                    linewidth=0.6, edgecolor=TEXT_SECONDARY, facecolor=tone, transform=ax.transAxes,
                                    zorder=1))
        head.set_position((x + padx, top - pady))
        body.set_position((x + padx, top - pady - hsize[1] - between))
        head.set_zorder(2)
        body.set_zorder(2)
        if x > 0:
            ax.add_patch(FancyArrowPatch((x - gap + 0.006, top - height / 2), (x - 0.006, top - height / 2),
                                         arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color=TEXT_SECONDARY,
                                         transform=ax.transAxes))
        x += width + gap


def draw_protocol(ax) -> None:
    """One patient: fine-tuning on the 14 days before the anchor, 84 test days, refits on the growing window."""
    ax.set_xlim(-6, 100)
    ax.set_ylim(-1.5, 6.2)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 4.2), 14, 0.9, facecolor="#cde2fb", edgecolor="none"))
    ax.text(7, 4.65, "14 days of history", ha="center", va="center", fontsize=PAPER_SMALL_PT)
    ax.add_patch(Rectangle((14, 4.2), 84, 0.9, facecolor="#f3f2ef", edgecolor=TEXT_SECONDARY, linewidth=0.4))
    ax.text(56, 4.65, "84 test days: a forecast every 30 min at 30, 60 and 120 min", ha="center", va="center",
            fontsize=PAPER_SMALL_PT)
    ax.plot([14, 14], [0.9, 5.6], color=TEXT_PRIMARY, linewidth=0.8)
    ax.text(14, 5.75, "anchor (day 14)", ha="center", va="bottom", fontsize=PAPER_SMALL_PT)
    texts = ("anchor fit: the population model fine-tuned on these 14 days",
             "refit 1 (day 28): every day so far", "refit 2 (day 42): every day so far")
    for k, (day, text) in enumerate(zip((14, 28, 42), texts)):
        y = 3.2 - k * 1.05
        ax.add_patch(Rectangle((0, y), day, 0.55, facecolor="#2a78d6", alpha=0.35 + 0.15 * k, edgecolor="none"))
        ax.add_patch(Rectangle((day, y), 14, 0.55, facecolor="none", edgecolor="#2a78d6", linewidth=0.5,
                               linestyle=(0, (2, 1))))
        ax.text(day + 15, y + 0.27, text, va="center", fontsize=PAPER_SMALL_PT)
    ax.text(-1, 2.0, "growing\nwindow", ha="right", va="center", fontsize=PAPER_SMALL_PT, color=TEXT_SECONDARY)
    ax.text(0, -1.3, "Each fit is used for the next k days (dashed): k = 1, 3, 7 or 14 days, here 14. Frozen: the "
            "anchor fit only.\nA refit starts again from the population model or continues the model in use "
            "(the mechanism).", fontsize=PAPER_SMALL_PT, color=TEXT_SECONDARY, va="bottom", linespacing=1.3)


# ============================================================================ figure 2


def fig02_population_personal(inputs: Inputs, settings: dict) -> Drawn:
    """a: the ladder from a personal-only model to the refitted one, per family; b: LightGBM personal only (from
    scratch) and with trees added to the population model, against the restart from the population model."""
    summary = inputs.csv("12_stage0_population_transfer", "model_summary")
    summary = points(summary[summary["period"] == "post_randomization"], ["skill_mean", "skill_low", "skill_high"])
    skill = inputs.csv("74_family_comparison", "skill")
    skill = points(skill[(skill["source"] == SCREEN) & (skill["state"] == "refitted")], ["skill", "low", "high"])
    weekly = inputs.csv("41_stage3_mechanism_analysis", "weekly_difference")
    weekly = points(weekly[(weekly["model"] == "lightgbm_ft") & weekly["policy"].isin(["scratch", "warm"])],
                    ["difference", "low", "high"])

    families = ["lightgbm", "lstm", "tsmixerx", "tide", "nhits", "patchtst"]
    steps = ["Personal\nonly", "Population", "Fine-tuned\n(14 days)", "Refitted"]
    fig = paper_figure(118)
    axes, strip = horizon_grid(fig, 2, heights=[1.35, 1])
    rows = []
    for column, horizon in enumerate(HORIZONS):
        ax = axes[0, column]
        frame = summary[summary["horizon_min"] == horizon].set_index("model")
        refit = skill[skill["horizon_min"] == horizon].set_index("model")
        for k, fam in enumerate(families):
            offset = (k - (len(families) - 1) / 2) * 0.06
            names = [fam, f"{fam}_pop", f"{fam}_ft"]
            ys = [(i, frame.loc[n, "skill_mean"], frame.loc[n, "skill_low"], frame.loc[n, "skill_high"])
                  for i, n in enumerate(names) if n in frame.index]
            if f"{fam}_ft" in refit.index:
                r = refit.loc[f"{fam}_ft"]
                ys.append((3, r["skill"], r["low"], r["high"]))
            if not ys:
                continue
            x = np.array([y[0] for y in ys]) + offset
            v = np.array([y[1] for y in ys])
            solid = x < 2.5
            ax.plot(x[solid], v[solid], color=colour(fam), linewidth=0.9, zorder=2)
            if (~solid).any() and solid.any():
                ax.plot([x[solid][-1], x[~solid][0]], [v[solid][-1], v[~solid][0]], color=colour(fam),
                        linewidth=0.9, linestyle=(0, (2, 1.2)), zorder=2)
            for (i, mean, low, high), xi in zip(ys, x):
                ax.plot([xi, xi], [low, high], color=colour(fam), linewidth=0.6, zorder=2)
                ax.plot(xi, mean, marker=marker(fam), color=colour(fam), markersize=3.4, zorder=3,
                        markerfacecolor=colour(fam), markeredgecolor="white", markeredgewidth=0.4)
                rows.append({"panel": "a", "horizon_min": horizon, "family": fam, "step": steps[i].replace("\n", " "),
                             "skill_pct": mean, "low": low, "high": high,
                             "source": "stage 3, 38 patients" if i == 3 else "stage 0, 40 patients"})
        if "arx" in frame.index:
            ax.axhline(frame.loc["arx", "skill_mean"], color=colour("arx"), linewidth=0.7, linestyle=(0, (4, 2)),
                       zorder=1)
            ax.text(3.3, frame.loc["arx", "skill_mean"] - 0.4, "ARX", color=colour("arx"),
                    fontsize=PAPER_SMALL_PT, va="top", ha="right")
            rows.append({"panel": "a", "horizon_min": horizon, "family": "arx", "step": "Personal only",
                         "skill_pct": frame.loc["arx", "skill_mean"], "low": frame.loc["arx", "skill_low"],
                         "high": frame.loc["arx", "skill_high"], "source": "stage 0, 40 patients"})
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps)
        ax.set_xlim(-0.45, 3.35)
        ax.grid(axis="x", visible=False)
        if column == 0:
            ax.set_ylabel("Skill over persistence (%)")

        ax = axes[1, column]
        for policy, style, text in (("scratch", "-", "Personal only, refitted"),
                                    ("warm", (0, (3, 1.5)), "Trees added to the model in use")):
            frame_w = weekly[(weekly["policy"] == policy) & (weekly["horizon_min"] == horizon)].sort_values("week")
            week = frame_w["week"] + 1
            ax.fill_between(week, frame_w["low"], frame_w["high"], color=colour("lightgbm"),
                            alpha=0.16 if policy == "scratch" else 0.08, linewidth=0)
            ax.plot(week, frame_w["difference"], color=colour("lightgbm"), linestyle=style, linewidth=1.0,
                    label=text)
            rows += [{"panel": "b", "horizon_min": horizon, "family": "lightgbm", "step": text, "week": int(w),
                      "difference_pts": d, "low": lo, "high": hi}
                     for w, d, lo, hi in zip(week, frame_w["difference"], frame_w["low"], frame_w["high"])]
        ax.axhline(0, **ZERO)
        ax.set_xlim(0.5, 12.5)
        ax.set_xticks([1, 4, 8, 12])
        if column == 1:
            ax.set_xlabel("Week after the anchor")
        if column == 0:
            ax.set_ylabel("LightGBM minus restart\nfrom population (points)")
            ax.legend(loc="lower right")
    hide_inner_y(axes)
    bottom_legend(strip, model_handles(families, line=True) + [Line2D([], [], color=colour("arx"), linewidth=0.7,
                                                                    linestyle=(0, (4, 2)), label="ARX")])
    notes = ["steps 1-3: stage 0 (12), 40 patients; refitted: stage 3 (74), 38 patients, each family with its "
             "recommended mechanism and interval (dashed segment)"]
    return Drawn(fig, pd.DataFrame(rows), notes)


# ============================================================================ figure 3


def fig03_families(inputs: Inputs, settings: dict) -> Drawn:
    """a, b: every learned family minus LightGBM, frozen and refitted, in the screening and the large cohorts; c:
    skill against the compute cost per test day, frozen and refitted, with the foundation models and their context."""
    pairs = inputs.csv("74_family_comparison", "pairwise")
    pairs = points(pairs, ["difference", "low", "high", "dmd"])
    context = points(inputs.csv("43_stage3_context", "context_skill"), ["skill", "low", "high"])
    cost = inputs.csv("43_stage3_context", "cost").set_index("forecaster")["seconds_per_day"]
    sizes = pairs.groupby("source")["patients"].max()
    sources = [s for s in pairs["source"].drop_duplicates() if s == SCREEN or sizes[s] >= settings["min_patients"]]
    others = ["lstm_ft", "tsmixerx_ft", "tide_ft"]

    fig = paper_figure(172)
    axes, strip = horizon_grid(fig, 3, heights=[1, 1, 1.05], sharey_rows=False, legend_rows=2)
    rows = []
    for r, state in enumerate(("frozen", "refitted")):
        for column, horizon in enumerate(HORIZONS):
            ax = axes[r, column]
            frame = versus(pairs[(pairs["state"] == state) & (pairs["horizon_min"] == horizon)], "lightgbm_ft")
            frame = frame[frame["model"].isin(others)]
            shown = forest(ax, sources, frame, others, "source")
            rows += [{"panel": "ab"[r], "state": state, **s} for s in shown.to_dict("records")]
            if column:
                ax.tick_params(labelleft=False)
            if column == 1:
                ax.set_xlabel(f"{state.capitalize()}: model minus LightGBM (points)")
        lo = min(a.get_xlim()[0] for a in axes[r])
        hi = max(a.get_xlim()[1] for a in axes[r])
        for ax in axes[r]:
            ax.set_xlim(lo, hi)

    learned = ["lightgbm_ft", "lstm_ft", "tsmixerx_ft", "tide_ft", "arx"]
    for column, horizon in enumerate(HORIZONS):
        ax = axes[2, column]
        frame = context[context["horizon_min"] == horizon].set_index("forecaster")
        for model in learned:
            names = [n for n in frame.index if n.startswith(model + " ")]
            frozen = [n for n in names if n.endswith(" never")]
            refit = [n for n in names if not n.endswith(" never")]
            if not frozen or not refit:
                continue
            xs = [cost[frozen[0]], cost[refit[0]]]
            ys = [frame.loc[frozen[0], "skill"], frame.loc[refit[0], "skill"]]
            ax.plot(xs, ys, color=colour(model), linewidth=0.7, zorder=2)
            ax.plot(xs[0], ys[0], marker=marker(model), color=colour(model), markerfacecolor="white", zorder=3,
                    linestyle="none")
            ax.plot(xs[1], ys[1], marker=marker(model), color=colour(model), zorder=3, linestyle="none")
            rows += [{"panel": "c", "horizon_min": horizon, "model": model, "state": s, "forecaster": n,
                      "seconds_per_day": x, "skill_pct": y} for s, n, x, y in
                     zip(("frozen", "refitted"), (frozen[0], refit[0]), xs, ys)]
        for model in ("chronos2", "timesfm"):
            names = sorted((n for n in frame.index if n.startswith(model + " ")),
                           key=lambda n: frame.loc[n, "context_days"])
            xs = [cost[n] for n in names]
            ys = [frame.loc[n, "skill"] for n in names]
            ax.plot(xs, ys, color=colour(model), linewidth=0.7, zorder=2)
            ax.plot(xs, ys, marker=marker(model), color=colour(model), linestyle="none", zorder=3,
                    markersize=3.0)
            ax.annotate("14 d", (xs[-1], ys[-1]), xytext=(2, -1), textcoords="offset points",
                        fontsize=PAPER_SMALL_PT, color=colour(model), va="top")
            rows += [{"panel": "c", "horizon_min": horizon, "model": model, "state": "context",
                      "forecaster": n, "seconds_per_day": x, "skill_pct": y} for n, x, y in zip(names, xs, ys)]
        ax.set_xscale("log")
        if column == 1:
            ax.set_xlabel("Compute per test day (s)")
        if column == 0:
            ax.set_ylabel("Skill (%)")
            ax.legend(handles=state_handles("frozen", "refitted"), loc="upper left")
    handles = model_handles(["lightgbm_ft", "lstm_ft", "tsmixerx_ft", "tide_ft", "arx", "chronos2", "timesfm"])
    handles += state_handles("a, b: tie or uncertain", "a, b: differs")
    bottom_legend(strip, handles, ncols=5)
    notes = [f"a-b: sources with at least {settings['min_patients']} patients; ARX and AR(3) lie 3-9 points below "
             "LightGBM and are left out of a-b (pairwise.csv of 74)",
             "c: stage 3 (38 patients); foundation models with 1, 3, 7 and 14 days of context, untrained"]
    return Drawn(fig, pd.DataFrame(rows), notes)


# ============================================================================ figure 4


def fig04_ageing(inputs: Inputs, settings: dict) -> Drawn:
    """a: weekly skill minus that of a model that cannot age (D101), frozen and refitted; b: the slope per week with
    its interval, and the slope of the mixed model of stage 1; c: recency against amount of data (stage 1)."""
    weekly = inputs.csv("43_stage3_context", "weekly_vs_context")
    slopes = inputs.csv("43_stage3_context", "ageing_vs_context")
    mixed = points(inputs.csv("21_stage1_ageing_analysis", "ageing_slopes"), ["slope_per_week", "slope_low",
                                                                             "slope_high"])
    effects = points(inputs.csv("21_stage1_ageing_analysis", "version_effects_mixed"), ["difference", "low", "high"])
    models = ["lightgbm_ft", "lstm_ft", "tsmixerx_ft", "tide_ft", "arx"]
    shown = ["lightgbm_ft", "lstm_ft"]
    reference = str(slopes["reference"].iloc[0])

    fig = paper_figure(138)
    axes, strip = horizon_grid(fig, 3, heights=[0.85, 1.05, 1.1], sharey_rows=False, legend_rows=2)
    rows = []
    for column, horizon in enumerate(HORIZONS):
        ax = axes[0, column]
        frame = weekly[weekly["horizon_min"] == horizon]
        for model in shown:
            for state, style in (("frozen", (0, (3, 1.5))), ("refitted", "-")):
                part = frame[(frame["model"] == model) & ((frame["policy"] == "never") == (state == "frozen"))]
                part = part.sort_values("week")
                week = part["week"] + 1
                ax.fill_between(week, part["low"], part["high"], color=colour(model), alpha=0.10, linewidth=0)
                ax.plot(week, part["difference"], color=colour(model), linestyle=style, linewidth=1.0)
                rows += [{"panel": "a", "horizon_min": horizon, "model": model, "state": state, "week": int(w),
                          "difference_pts": d, "low": lo, "high": hi}
                         for w, d, lo, hi in zip(week, part["difference"], part["low"], part["high"])]
        ax.set_xlim(0.5, 12.5)
        ax.set_xticks([1, 4, 8, 12])
        if column == 1:
            ax.set_xlabel("Week after the anchor")
        if column == 0:
            ax.set_ylabel(f"Skill minus {label(reference.split()[0])},\n{reference.split(' ', 1)[1]} context (points)")
            ax.legend(handles=[Line2D([], [], color=TEXT_SECONDARY, linestyle=(0, (3, 1.5)), label="frozen"),
                               Line2D([], [], color=TEXT_SECONDARY, label="refitted")], loc="lower right")
    lo = min(a.get_ylim()[0] for a in axes[0])
    hi = max(a.get_ylim()[1] for a in axes[0])
    for ax in axes[0]:
        ax.set_ylim(lo, hi)
    hide_inner_y(axes[:1])

    for column, horizon in enumerate(HORIZONS):
        ax = axes[1, column]
        frame = slopes[slopes["horizon_min"] == horizon]
        names = models + ["timesfm"]
        for i, model in enumerate(names):
            y = len(names) - 1 - i
            part = frame[frame["model"] == model]
            frozen = part[part["policy"].isin(["never", "context"])]
            if model == "timesfm":
                frozen = frozen[frozen["label"].str.endswith(" 14 d")]
            refit = part[~part["policy"].isin(["never", "context"])]
            for kind, sub, dy, filled in (("frozen", frozen, 0.14, False), ("refitted", refit, -0.14, True)):
                if sub.empty:
                    continue
                r = sub.iloc[0]
                ax.plot([r["low"], r["high"]], [y + dy, y + dy], color=colour(model), linewidth=0.8)
                ax.plot(r["slope_per_week"], y + dy, marker=marker(model), color=colour(model), linestyle="none",
                        markerfacecolor=colour(model) if filled else "white", markersize=3.6)
                rows.append({"panel": "b", "horizon_min": horizon, "model": model, "state": kind,
                             "estimator": f"against {reference}", "slope_pts_per_week": r["slope_per_week"],
                             "low": r["low"], "high": r["high"]})
            stage1 = mixed[(mixed["model"] == model) & (mixed["horizon_min"] == horizon)]
            if not stage1.empty:
                r = stage1.iloc[0]
                ax.plot([r["slope_low"], r["slope_high"]], [y + 0.36, y + 0.36], color=TEXT_SECONDARY,
                        linewidth=0.5)
                ax.plot(r["slope_per_week"], y + 0.36, marker="|", color=TEXT_SECONDARY, markersize=4,
                        linestyle="none")
                rows.append({"panel": "b", "horizon_min": horizon, "model": model, "state": "frozen",
                             "estimator": "mixed model, stage 1", "slope_pts_per_week": r["slope_per_week"],
                             "low": r["slope_low"], "high": r["slope_high"]})
        ax.axvline(0, **ZERO)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([label(m) + (" (control)" if m == "timesfm" else "") for m in names][::-1])
        ax.set_ylim(-0.6, len(names) - 0.3)
        ax.grid(axis="y", visible=False)
        if column:
            ax.tick_params(labelleft=False)
        if column == 1:
            ax.set_xlabel("Change in skill per week (points)")
    same_limits(axes[1], "x")

    order = ["lightgbm_ft", "lstm_ft", "tsmixerx_ft", "tide_ft", "nhits_ft", "patchtst_ft", "arx", "ar3"]
    kinds = [("recency (rolling - frozen)", 0.45, "Recency: last 14 days\nminus frozen"),
             ("data (growing - rolling)", 1.0, "Amount of data: growing\nminus last 14 days")]
    for column, horizon in enumerate(HORIZONS):
        ax = axes[2, column]
        frame = effects[effects["horizon_min"] == horizon]
        names = [m for m in order if m in set(frame["model"])]
        for i, model in enumerate(names):
            y = len(names) - 1 - i
            for k, (effect, alpha, text) in enumerate(kinds):
                hit = frame[(frame["model"] == model) & (frame["effect"] == effect)]
                if hit.empty:
                    continue
                r = hit.iloc[0]
                dy = 0.19 - 0.38 * k
                ax.barh(y + dy, r["difference"], height=0.34, color=colour(model), alpha=alpha, linewidth=0)
                ax.plot([r["low"], r["high"]], [y + dy, y + dy], color=TEXT_PRIMARY, linewidth=0.5)
                rows.append({"panel": "c", "horizon_min": horizon, "model": model, "effect": effect,
                             "difference_pts": r["difference"], "low": r["low"], "high": r["high"],
                             "verdict": r["verdict"]})
        ax.axvline(0, **ZERO)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([label(m) for m in names][::-1])
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.grid(axis="y", visible=False)
        if column:
            ax.tick_params(labelleft=False)
        if column == 1:
            ax.set_xlabel("Effect on skill (points), stage 1")
    same_limits(axes[2], "x")
    extra = state_handles("b: frozen", "b: refitted")
    extra += [Line2D([], [], color=TEXT_SECONDARY, marker="|", linestyle="none", markersize=4,
                     label="b: frozen, stage 1 mixed model")]
    extra += [Rectangle((0, 0), 1, 1, color=TEXT_SECONDARY, alpha=a, linewidth=0, label="c: " + t.replace("\n", " "))
              for _, a, t in kinds]
    bottom_legend(strip, model_handles(order + ["timesfm"]) + extra, ncols=7)
    notes = [f"a, b: stage 3 (43), against {reference} on the same patients and days (D101); b: patients with at "
             "least 6 weeks; stage 1 slopes from 21 (frozen models, mixed model, D78); c: stage 1 (21, D78)"]
    return Drawn(fig, pd.DataFrame(rows), notes)


# ============================================================================ figure 5

WINDOWS = {"growing": "Growing", "exp14": "Half-life 14 d", "rolling28": "Last 28 d", "rolling14": "Last 14 d",
           "rolling7": "Last 7 d"}
INTERVALS = (1, 3, 7, 14)
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    "paper_diverging", ["#a3282a", "#e57f7c", "#f0efec", "#6da7ec", "#184f95"])


def fig05_policy(inputs: Inputs, settings: dict) -> Drawn:
    """a: the gain over never of every window and interval (stage 2); b: the share of the daily refit's gain each
    interval keeps on the growing window; c: gain against compute, with the drift triggers."""
    grid = points(inputs.csv("31_stage2_policy_analysis", "later_anchor_gains"), ["skill_difference", "low", "high"])
    grid = grid[(grid["anchor_day"] == 14) & (grid["against"] == "never")]
    intervals = points(inputs.csv("31_stage2_policy_analysis", "interval_comparison"), ["share_of_daily_gain"])
    table = inputs.csv("31_stage2_policy_analysis", "policy_table")
    models = ["lightgbm_ft", "lstm_ft", "tsmixerx_ft", "tide_ft", "arx"]

    fig = paper_figure(184)
    axes, strip = horizon_grid(fig, 3, heights=[2.2, 1.35, 1.45], sharey_rows=False)
    rows = []
    entries = [(m, w) for m in models for w in WINDOWS]
    limit = np.nanmax(np.abs(grid[grid["model"].isin(models)]["skill_difference"])) if not grid.empty else 1.0
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
    for column, horizon in enumerate(HORIZONS):
        ax = axes[0, column]
        frame = grid[grid["horizon_min"] == horizon].set_index("policy")
        values = np.full((len(entries), len(INTERVALS)), np.nan)
        for i, (model, window) in enumerate(entries):
            part = frame[frame["model"] == model]
            for j, k in enumerate(INTERVALS):
                name = f"{window}_every{k}"
                if name in part.index:
                    r = part.loc[name]
                    values[i, j] = r["skill_difference"]
                    rows.append({"panel": "a", "horizon_min": horizon, "model": model, "window": window,
                                 "interval_days": k, "gain_pts": r["skill_difference"], "low": r["low"],
                                 "high": r["high"], "verdict": r["verdict"]})
        ax.imshow(np.ma.masked_invalid(values), cmap=DIVERGING, norm=norm, aspect="auto",
                  interpolation="nearest")
        for i in range(len(entries)):
            for j in range(len(INTERVALS)):
                v = values[i, j]
                if np.isfinite(v):
                    rgb = DIVERGING(norm(v))[:3]
                    dark = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 0.5
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=PAPER_MIN_PT,
                            color="white" if dark else TEXT_PRIMARY)
        for boundary in range(len(WINDOWS), len(entries), len(WINDOWS)):
            ax.axhline(boundary - 0.5, color="white", linewidth=1.6)
        ax.set_xticks(range(len(INTERVALS)))
        ax.set_xticklabels([str(k) for k in INTERVALS])
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels([WINDOWS[w] for _, w in entries])
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if column == 1:
            ax.set_xlabel("Refit every k days")
        if column:
            ax.tick_params(labelleft=False)
        else:
            for k, model in enumerate(models):
                ax.annotate(label(model), xy=(0, (k + 0.5) * len(WINDOWS) - 0.5), xycoords=("axes fraction", "data"),
                            xytext=(-50, 0), textcoords="offset points", rotation=90, ha="center", va="center",
                            fontweight="bold", color=colour(model) if model != "tide_ft" else "#9c6a00")

    for column, horizon in enumerate(HORIZONS):
        ax = axes[1, column]
        frame = intervals[(intervals["horizon_min"] == horizon) & intervals["policy"].str.startswith("growing_every")]
        for model in models:
            part = frame[frame["model"] == model].sort_values("interval_days")
            if part.empty:
                continue
            ax.plot(part["interval_days"], part["share_of_daily_gain"], color=colour(model), linewidth=0.8)
            for _, r in part.iterrows():
                worse = r["verdict_to_daily"] == "worse"
                ax.plot(r["interval_days"], r["share_of_daily_gain"], marker=marker(model), color=colour(model),
                        markerfacecolor=colour(model) if worse else "white", markersize=3.6, linestyle="none")
                rows.append({"panel": "b", "horizon_min": horizon, "model": model,
                             "interval_days": r["interval_days"], "share_of_daily_gain_pct": r["share_of_daily_gain"],
                             "difference_to_daily": r["difference_to_daily"], "verdict": r["verdict_to_daily"]})
        ax.set_xscale("log", base=2)
        ax.set_xticks(INTERVALS)
        ax.set_xticklabels([str(k) for k in INTERVALS])
        ax.minorticks_off()
        shares = intervals[intervals["model"].isin(models)]["share_of_daily_gain"]
        ax.set_ylim(min(60, 10 * np.floor((shares.min() - 5) / 10)) if shares.notna().any() else 40, 103)
        if column == 1:
            ax.set_xlabel("Refit every k days (growing window)")
        if column == 0:
            ax.set_ylabel("Daily refit's gain\nkept (%)")
            ax.legend(handles=state_handles("tie with daily", "below daily"), loc="lower left")
    hide_inner_y(axes[1:2])

    fixed = [f"growing_every{k}" for k in sorted(INTERVALS, reverse=True)]
    for column, horizon in enumerate(HORIZONS):
        ax = axes[2, column]
        key = f"skill_{horizon}"
        for model in models:
            part = table[table["model"] == model].set_index("policy")
            if "never" not in part.index:
                continue
            base = part.loc["never", key]
            line = [p for p in ["never"] + fixed if p in part.index]
            xs = [max(part.loc[p, "seconds_per_day"], 1e-4) for p in line]
            ys = [(part.loc[p, key] - base) * 100 for p in line]
            ax.plot(xs, ys, color=colour(model), linewidth=0.8, marker=marker(model), markersize=3.0,
                    markerfacecolor=colour(model))
            for p in [p for p in part.index if "page_hinkley" in p and not p.startswith("even")]:
                ax.plot(max(part.loc[p, "seconds_per_day"], 1e-4), (part.loc[p, key] - base) * 100,
                        marker=marker(model), color=colour(model), markerfacecolor="white", markersize=3.6,
                        linestyle="none")
                line = line + [p]
            rows += [{"panel": "c", "horizon_min": horizon, "model": model, "policy": p,
                      "seconds_per_day": part.loc[p, "seconds_per_day"],
                      "gain_pts": (part.loc[p, key] - base) * 100, "front": part.loc[p, "front"]} for p in line]
        ax.axhline(0, **ZERO)
        ax.set_xscale("log")
        if column == 1:
            ax.set_xlabel("Compute per test day (s)")
        if column == 0:
            ax.set_ylabel("Gain over frozen\n(points)")
            ax.legend(handles=state_handles("drift trigger (Page-Hinkley)", "every 14, 7, 3, 1 days"),
                      loc="upper left")
    same_limits(axes[2])
    hide_inner_y(axes[2:])
    axes[0, 0].figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=DIVERGING), ax=axes[0, :], location="right",
                 shrink=0.5, pad=0.01, aspect=30, label="Gain over frozen (points)")
    bottom_legend(strip, model_handles(models))
    notes = ["stage 2 (30, 31), anchor day 14, 38 patients; a: every window and interval; b: open = ties with "
             "the daily refit (D75), filled = below it; c: fixed intervals on the growing window and the "
             "Page-Hinkley trigger"]
    return Drawn(fig, pd.DataFrame(rows), notes)


# ============================================================================ figure 6


def fig06_data(inputs: Inputs, settings: dict) -> Drawn:
    """a: the gain of refitting over frozen week by week (screening, each model at its interval); b: the same in Loop
    for 36 weeks (confirmation, recommended arm); c: the gain with more history at the anchor (days 14, 42, 70)."""
    weekly = points(inputs.csv("31_stage2_policy_analysis", "weekly_gain"), ["gain_mean", "gain_low", "gain_high"])
    optimal = inputs.csv("31_stage2_policy_analysis", "optimal_interval")
    loop = points(inputs.csv("51_confirmation_analysis", "weekly_gain"), ["gain_mean", "gain_low", "gain_high"])
    anchors = points(inputs.csv("31_stage2_policy_analysis", "later_anchor_gains"), ["skill_difference", "low",
                                                                                     "high"])
    models = ["lightgbm_ft", "lstm_ft", "tsmixerx_ft", "tide_ft", "arx"]
    interval = optimal.drop_duplicates("model").set_index("model")["model_interval_days"]
    recommended = {"lightgbm_ft": "restart_every3", "lstm_ft": "warm_every3", "tsmixerx_ft": "restart_other_every14",
                   "tide_ft": "restart_every14", "arx": "online_l1"}

    fig = paper_figure(150)
    axes, strip = horizon_grid(fig, 3, heights=[1, 1, 0.9], sharey_rows=False)
    rows = []
    for column, horizon in enumerate(HORIZONS):
        for r, (frame, source) in enumerate(((weekly, "screen"), (loop, "loop/closed_insulin"))):
            ax = axes[r, column]
            for model in models:
                if source == "screen":
                    if model not in interval.index:
                        continue
                    policy = f"growing_every{int(interval[model])}"
                    part = frame[(frame["model"] == model) & (frame["policy"] == policy)
                                 & (frame["anchor_day"] == 14)]
                else:
                    policy = recommended[model]
                    part = frame[(frame["cohort"] == source) & (frame["model"] == model)
                                 & (frame["policy"] == policy)]
                part = part[part["horizon_min"] == horizon].sort_values("week")
                if part.empty:
                    continue
                week = part["week"] + 1
                ax.fill_between(week, part["gain_low"], part["gain_high"], color=colour(model), alpha=0.06,
                                linewidth=0)
                ax.plot(week, part["gain_mean"], color=colour(model), linewidth=0.9)
                rows += [{"panel": "ab"[r], "source": source, "horizon_min": horizon, "model": model,
                          "policy": policy, "week": int(w), "gain_pts": g, "low": lo, "high": hi}
                         for w, g, lo, hi in zip(week, part["gain_mean"], part["gain_low"], part["gain_high"])]
            ax.axhline(0, **ZERO)
            ax.set_xlim(0.5, 12.5 if r == 0 else 36.5)
            ax.set_xticks([1, 4, 8, 12] if r == 0 else [1, 12, 24, 36])
            if column == 1:
                ax.set_xlabel("Week after the anchor" + (" (screening)" if r == 0 else " (Loop, confirmation)"))
            if column == 0:
                ax.set_ylabel("Gain over frozen\n(points)")

        ax = axes[2, column]
        frame = anchors[(anchors["horizon_min"] == horizon) & (anchors["policy"] == "growing_every7")
                        & (anchors["against"] == "never")]
        for model in models:
            part = frame[frame["model"] == model].sort_values("anchor_day")
            if part.empty:
                continue
            ax.plot(part["anchor_day"], part["skill_difference"], color=colour(model), linewidth=0.8,
                    marker=marker(model), markersize=3.4)
            for _, r in part.iterrows():
                ax.plot([r["anchor_day"]] * 2, [r["low"], r["high"]], color=colour(model), linewidth=0.5)
            rows += [{"panel": "c", "horizon_min": horizon, "model": model, "anchor_day": int(r["anchor_day"]),
                      "patients": int(r["patients"]), "gain_pts": r["skill_difference"], "low": r["low"],
                      "high": r["high"]} for _, r in part.iterrows()]
        ax.axhline(0, **ZERO)
        ax.set_xticks([14, 42, 70])
        ax.set_xlim(8, 76)
        if column == 1:
            ax.set_xlabel("Days of history at the anchor (refit every 7 days, 84 test days)")
        if column == 0:
            ax.set_ylabel("Gain over frozen\n(points)")
    for row in axes:
        same_limits(row)
    hide_inner_y(axes)
    bottom_legend(strip, model_handles(models, line=True))
    notes = ["a: stage 2, growing window at each model's interval (31: 3 days for LightGBM and the LSTM, 14 for "
             "TiDE and TSMixerx, 7 for ARX); b: Loop, 120 patients, each model's recommended arm (51); c: stage 2 "
             "(31), anchor day 14 on 38 patients, days 42 and 70 on the patients with that much history"]
    return Drawn(fig, pd.DataFrame(rows), notes)


# ============================================================================ figure 7

MECHANISMS = {"warm": "continued", "scratch": "personal only (from scratch)", "leaves_d0.9": "leaves, decay 0.9",
              "leaves_d0.5": "leaves, decay 0.5", "leaves_d0": "leaves re-estimated",
              "online_l0.9995": "forgetting 0.9995", "online_l0.999": "forgetting 0.999",
              "online_l0.995": "forgetting 0.995", "online_l0.99": "forgetting 0.99"}


def fig07_mechanism(inputs: Inputs, settings: dict) -> Drawn:
    """a: each mechanism against restarting from the population model (stage 3; ARX: forgetting against none);
    b: the skill of the untrained foundation models with more context, against LightGBM and ARX; c: the IS-LSTM of
    Shen and Kleinberg against frozen, the restart every 3 days and the recommended continuation (75)."""
    mech = points(inputs.csv("41_stage3_mechanism_analysis", "vs_reference"), ["skill_difference", "low", "high"])
    chosen = inputs.yaml("11_pretrain_population", "chosen_settings.yaml")
    context = points(inputs.csv("43_stage3_context", "context_skill"), ["skill", "low", "high"])
    precedent = points(inputs.csv("75_precedent_is_lstm", "versus"), ["skill_difference", "low", "high"])

    def name(model: str, policy: str) -> str:
        if policy == "restart_other":
            train = chosen.get(family(model), {}).get("finetune", {}).get("train", "all")
            return f"{label(model)}: restart, {'all layers' if train == 'head' else 'output layers'}"
        return f"{label(model)}: {MECHANISMS.get(policy, policy)}"

    order = [("lightgbm_ft", p) for p in ("warm", "leaves_d0.9", "leaves_d0.5", "leaves_d0", "scratch")]
    order += [(m, p) for m in ("lstm_ft", "tsmixerx_ft", "tide_ft") for p in ("warm", "restart_other")]
    order += [("arx", p) for p in ("online_l0.9995", "online_l0.999")]
    fig = paper_figure(154)
    axes, strip = horizon_grid(fig, 3, heights=[1.55, 0.85, 0.95], sharey_rows=False, legend_rows=3)
    rows = []
    for column, horizon in enumerate(HORIZONS):
        ax = axes[0, column]
        frame = mech[mech["horizon_min"] == horizon]
        for i, (model, policy) in enumerate(order):
            hit = frame[(frame["model"] == model) & (frame["policy"] == policy)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            y = len(order) - 1 - i
            full = r["verdict"] in ("better", "worse")
            ax.plot([r["low"], r["high"]], [y, y], color=colour(model), linewidth=0.8)
            ax.plot(r["skill_difference"], y, marker=marker(model), color=colour(model), linestyle="none",
                    markerfacecolor=colour(model) if full else "white", markersize=3.6)
            rows.append({"panel": "a", "horizon_min": horizon, "model": model, "mechanism": policy,
                         "against": r["against"], "difference_pts": r["skill_difference"], "low": r["low"],
                         "high": r["high"], "verdict": r["verdict"]})
        ax.axvline(0, **ZERO)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([name(m, p) for m, p in order][::-1])
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.grid(axis="y", visible=False)
        if column:
            ax.tick_params(labelleft=False)
        if column == 1:
            ax.set_xlabel("Mechanism minus restart from the population model (points; ARX: minus no forgetting)")
    same_limits(axes[0], "x")

    references = (("lightgbm_ft restart", "-", "LightGBM, refitted"),
                  ("lightgbm_ft never", (0, (3, 1.5)), "LightGBM, frozen"),
                  ("arx online_l1", (0, (1, 1)), "ARX, every sample"))
    for column, horizon in enumerate(HORIZONS):
        ax = axes[1, column]
        frame = context[context["horizon_min"] == horizon]
        for model in ("chronos2", "timesfm"):
            part = frame[frame["model"] == model].sort_values("context_days")
            ax.plot(part["context_days"], part["skill"], color=colour(model), marker=marker(model), markersize=3.4)
            for _, r in part.iterrows():
                ax.plot([r["context_days"]] * 2, [r["low"], r["high"]], color=colour(model), linewidth=0.5)
            rows += [{"panel": "b", "horizon_min": horizon, "model": model, "context_days": r["context_days"],
                      "skill_pct": r["skill"], "low": r["low"], "high": r["high"]} for _, r in part.iterrows()]
        for forecaster, style, text in references:
            hit = frame[frame["forecaster"] == forecaster]
            if hit.empty:
                continue
            value = hit.iloc[0]["skill"]
            model = forecaster.split()[0]
            ax.axhline(value, color=colour(model), linestyle=style, linewidth=0.8)
            rows.append({"panel": "b", "horizon_min": horizon, "model": forecaster, "skill_pct": value})
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 3, 7, 14])
        ax.set_xticklabels(["1", "3", "7", "14"])
        ax.minorticks_off()
        ax.set_xlim(0.8, 17)
        if column == 1:
            ax.set_xlabel("Days of context read by the untrained foundation model")
        if column == 0:
            ax.set_ylabel("Skill (%)")
    same_limits(axes[1])
    hide_inner_y(axes[1:2])

    cohorts = [("replace_bg/core", "lstm_ft"), ("dclp3/clc_closed_insulin", "lstm_ft"),
               ("loop/closed_insulin", "lstm_ft"), ("dclp3/sap_open_glucose", "lstm_ft_noexog")]
    comparisons = [("against never", "frozen", "o"), ("against the restart", "restart every 3 days", "v"),
                   ("against the recommended mechanism", "continued every 3 days (recommended)", "d")]
    for column, horizon in enumerate(HORIZONS):
        ax = axes[2, column]
        frame = precedent[(precedent["horizon_min"] == horizon) & (precedent["policy"] == "incremental_w7_every7")]
        for i, (cohort, model) in enumerate(cohorts):
            for k, (comparison, text, shape) in enumerate(comparisons):
                hit = frame[(frame["cohort"] == cohort) & (frame["model"] == model)
                            & (frame["comparison"] == comparison)]
                if hit.empty:
                    continue
                r = hit.iloc[0]
                y = len(cohorts) - 1 - i + 0.22 - 0.22 * k
                full = r["verdict"] in ("better", "worse")
                ax.plot([r["low"], r["high"]], [y, y], color=colour("lstm"), linewidth=0.8)
                ax.plot(r["skill_difference"], y, marker=shape, color=colour("lstm"), linestyle="none",
                        markerfacecolor=colour("lstm") if full else "white", markersize=3.6)
                rows.append({"panel": "c", "horizon_min": horizon, "cohort": cohort, "model": model,
                             "comparison": f"IS-LSTM minus {text}", "difference_pts": r["skill_difference"],
                             "low": r["low"], "high": r["high"], "verdict": r["verdict"],
                             "patients": int(r["patients"])})
        ax.axvline(0, **ZERO)
        ax.set_yticks(range(len(cohorts)))
        ax.set_yticklabels([COHORT_LABELS[c] for c, _ in cohorts][::-1])
        ax.set_ylim(-0.6, len(cohorts) - 0.4)
        ax.grid(axis="y", visible=False)
        if column:
            ax.tick_params(labelleft=False)
        if column == 1:
            ax.set_xlabel("IS-LSTM (last 7 days, every 7 days) minus the LSTM (points)")
    same_limits(axes[2], "x")
    handles = model_handles(["lightgbm_ft", "lstm_ft", "tsmixerx_ft", "tide_ft", "arx", "chronos2", "timesfm"])
    handles += state_handles("a, c: tie or uncertain", "a, c: differs")
    handles += [Line2D([], [], color=colour(f.split()[0]), linestyle=s, linewidth=0.8, label=f"b: {t}")
                for f, s, t in references]
    handles += [Line2D([], [], color=colour("lstm"), marker=s, linestyle="none", markersize=3.6,
                       markerfacecolor="white", label=f"c: minus {t}") for _, t, s in comparisons]
    bottom_legend(strip, handles, ncols=6)
    notes = ["a: stage 3 (41), 38 patients; ARX with forgetting 0.995 and 0.99 (5 to 16 points below) left out; "
             "restart_other is the fine-tuning choice script 11 did not take "
             "(chosen_settings.yaml); b: stage 3 (43); c: confirmation cohorts (75), the LSTM with insulin and "
             "carbohydrates, its glucose-only twin in DCLP3 SAP; parameter transfer ties with IS-LSTM (versus.csv)"]
    return Drawn(fig, pd.DataFrame(rows), notes)


# ============================================================================ figure 8


def fig08_inputs(inputs: Inputs, settings: dict) -> Drawn:
    """a: the gain of refitting with insulin and carbohydrates against the same model with glucose only, in the
    screening and the large cohorts; b: what wearables add, patient by patient (73)."""
    screen = points(inputs.csv("31_stage2_policy_analysis", "exog_refit"), ["gain_all_inputs", "gain_glucose_only"])
    optimal = inputs.csv("31_stage2_policy_analysis", "optimal_interval")
    confirm = points(inputs.csv("51_confirmation_analysis", "exog_confirmation"),
                     ["gain_all_inputs", "gain_glucose_only"])
    wear = points(inputs.csv("73_wearables", "per_patient"), ["difference"])
    models = ["lightgbm_ft", "lstm_ft", "tsmixerx_ft", "tide_ft", "arx"]
    interval = optimal.drop_duplicates("model").set_index("model")["model_interval_days"]
    recommended = {"lightgbm_ft": "restart_every3", "lstm_ft": "warm_every3", "tsmixerx_ft": "restart_other_every14",
                   "tide_ft": "restart_every14", "arx": "online_l1"}
    sources = [("screen", "o"), ("replace_bg/core", "s"), ("dclp3/clc_closed_insulin", "^"),
               ("loop/closed_insulin", "D")]

    fig = paper_figure(124)
    axes, strip = horizon_grid(fig, 2, heights=[1.15, 1], sharey_rows=False, legend_rows=1)
    rows = []
    for column, horizon in enumerate(HORIZONS):
        ax = axes[0, column]
        for source, shape in sources:
            for model in models:
                if source == "screen":
                    policy = "online" if model == "arx" else f"growing_every{int(interval.get(model, 7))}"
                    hit = screen[(screen["model"] == model) & (screen["policy"] == policy)]
                else:
                    policy = recommended[model]
                    hit = confirm[(confirm["cohort"] == source) & (confirm["model"] == model)
                                  & (confirm["policy"] == policy)]
                hit = hit[hit["horizon_min"] == horizon]
                if hit.empty:
                    continue
                r = hit.iloc[0]
                ax.plot(r["gain_glucose_only"], r["gain_all_inputs"], marker=shape, color=colour(model),
                        linestyle="none", markersize=3.6, markeredgecolor="white", markeredgewidth=0.4)
                rows.append({"panel": "a", "horizon_min": horizon, "source": source, "model": model,
                             "policy": policy, "gain_all_inputs_pts": r["gain_all_inputs"],
                             "gain_glucose_only_pts": r["gain_glucose_only"], "difference": r["difference"],
                             "low": r["low"], "high": r["high"], "verdict": r["verdict"]})
        top = max(ax.get_xlim()[1], ax.get_ylim()[1], 0.5)
        line = np.array([0, top])
        ax.plot(line, line, color=TEXT_SECONDARY, linewidth=0.6, linestyle=(0, (3, 1.5)))
        ax.plot(line / 2, line, color=TEXT_SECONDARY, linewidth=0.6, linestyle=(0, (1, 1)))
        ax.set_xlim(min(0, ax.get_xlim()[0]), top)
        ax.set_ylim(min(0, ax.get_ylim()[0]), top)
        if column == len(HORIZONS) - 1:
            ax.text(top * 0.97, top * 0.97, "same gain", ha="right", va="top", fontsize=PAPER_SMALL_PT,
                    rotation=0, color=TEXT_SECONDARY)
            ax.text(top * 0.5, top * 0.97, "twice", ha="right", va="top", fontsize=PAPER_SMALL_PT,
                    color=TEXT_SECONDARY)
        if column == 1:
            ax.set_xlabel("Gain of refitting over frozen, glucose only (points)")
        if column == 0:
            ax.set_ylabel("Gain with insulin and\ncarbohydrates (points)")

    groups = [("lightgbm_ft", "restart_every3"), ("lightgbm_ft_noexog", "restart_every3"),
              ("arx", "online_l1"), ("arx_noexog", "online_l1")]
    rng = np.random.default_rng(0)
    for column, horizon in enumerate(HORIZONS):
        ax = axes[1, column]
        frame = wear[wear["horizon_min"] == horizon]
        for g, (model, refit) in enumerate(groups):
            for k, (policy, filled) in enumerate((("never", False), (refit, True))):
                part = frame[(frame["model"] == model) & (frame["policy"] == policy)]
                if part.empty:
                    continue
                x = g + (k - 0.5) * 0.36
                jitter = rng.uniform(-0.08, 0.08, len(part))
                ax.plot(x + jitter, part["difference"], marker=marker(model), color=colour(model), linestyle="none",
                        markersize=2.6, alpha=0.75, markerfacecolor=colour(model) if filled else "white",
                        markeredgewidth=0.5)
                ax.plot([x - 0.14, x + 0.14], [part["difference"].mean()] * 2, color=TEXT_PRIMARY, linewidth=1.0)
                rows += [{"panel": "b", "horizon_min": horizon, "model": model, "policy": policy,
                          "cohort": r["cohort"], "patient_id": r["patient_id"], "difference_pts": r["difference"]}
                         for _, r in part.iterrows()]
        ax.axhline(0, **ZERO)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([label(m).replace(", glucose only", "\nglucose only") for m, _ in groups])
        ax.grid(axis="x", visible=False)
        if column == 0:
            ax.set_ylabel("Wearables minus without\n(points, per patient)")
            ax.legend(handles=state_handles("frozen", "refitted"), loc="upper left")
    hide_inner_y(axes[1:])
    for ax in axes[1, 1:]:
        ax.sharey(axes[1, 0])
    handles = model_handles(models) + [Line2D([], [], color=TEXT_SECONDARY, marker=s, linestyle="none",
                                              markersize=3.6, label=f"a: {COHORT_LABELS[c]}") for c, s in sources]
    bottom_legend(strip, handles, ncols=9)
    notes = ["a: the gain over frozen with all inputs against the glucose-only twin's gain, screening (31, each "
             "model's interval on the growing window; ARX at every sample) and confirmation (51, recommended arm); "
             "b: patients with wearables (73), LightGBM refitted every 3 days and ARX at every sample; the line "
             "is the mean"]
    return Drawn(fig, pd.DataFrame(rows), notes)


# ============================================================================ figure 9


def fig09_replication(inputs: Inputs, settings: dict) -> Drawn:
    """The gain of refitting at each model's optimal interval over frozen, in the screening and in every
    confirmation cohort with enough patients."""
    screen = points(inputs.csv("31_stage2_policy_analysis", "interval_comparison"),
                    ["gain_over_never", "gain_low", "gain_high"])
    optimal = inputs.csv("31_stage2_policy_analysis", "optimal_interval")
    replication = points(inputs.csv("51_confirmation_analysis", "replication"), ["skill_difference", "low", "high"])
    interval = optimal.drop_duplicates("model").set_index("model")["model_interval_days"]
    replication = replication[replication["finding"] == "gain_over_never"]
    sizes = replication.groupby("cohort")["patients"].max()
    cohorts = [c for c in replication["cohort"].drop_duplicates() if sizes[c] >= settings["small_cohorts"]]
    models = ["lightgbm_ft", "lstm_ft", "tsmixerx_ft", "tide_ft", "arx", "ar3"]

    frames = []
    for model in models:
        part = screen[(screen["model"] == model) & (screen["policy"] == f"growing_every{int(interval.get(model, 7))}")]
        frames.append(part.assign(cohort="screen", difference=part["gain_over_never"], low=part["gain_low"],
                                  high=part["gain_high"], verdict=part["verdict_over_never"]))
    for model in models:
        for twin in (model, f"{model}_noexog" if model != "ar3" else model):
            part = replication[replication["model"] == twin]
            part = part[part["cohort"].isin(cohorts)]
            if twin != model:
                part = part[part["cohort"].str.endswith("_glucose")]
            elif model != "ar3":
                part = part[~part["cohort"].str.endswith("_glucose")]
            frames.append(part.assign(model=model, entry=twin, difference=part["skill_difference"]))
    data = pd.concat(frames, ignore_index=True)

    fig = paper_figure(118)
    axes, strip = horizon_grid(fig, 1, letters=[""], sharey_rows=False)
    rows = ["screen"] + cohorts
    for column, horizon in enumerate(HORIZONS):
        ax = axes[0, column]
        frame = data[data["horizon_min"] == horizon]
        forest(ax, rows, frame, models, "cohort", spread=0.3)
        ax.axhline(len(rows) - 1.5, color=TEXT_SECONDARY, linewidth=0.4, linestyle=(0, (2, 2)))
        if column:
            ax.tick_params(labelleft=False)
        if column == 1:
            ax.set_xlabel("Gain of refitting at the optimal interval over frozen (points)")
    same_limits(axes[0], "x")
    top = float(np.ceil(data["difference"].max() + 1.0))  # an interval of a small cohort must not flatten the rest
    cut = sorted(data.loc[data["high"] > top, "cohort"].map(lambda c: COHORT_LABELS.get(c, c)).unique())
    if cut:
        for ax in axes[0]:
            ax.set_xlim(ax.get_xlim()[0], top)
    labels = [COHORT_LABELS.get(c, c) + (f" ({int(sizes[c])})" if c in sizes else " (38)") for c in rows]
    axes[0, 0].set_yticklabels(labels[::-1])
    bottom_legend(strip, model_handles(models) + state_handles("tie or uncertain", "differs"), ncols=8)
    keep = ["cohort", "model", "entry", "horizon_min", "policy", "patients", "difference", "low", "high", "verdict",
            "screen_difference", "screen_verdict", "replicated"]
    notes = [f"screening: stage 2 (31), growing window at each model's interval; cohorts with at least "
             f"{settings['small_cohorts']} patients (51), each model refitted from the population model at its "
             "interval; glucose-only cohorts: the twins; the number of patients in brackets"]
    if cut:
        notes.append(f"intervals cut at {top:g} points: {', '.join(cut)} (the data file has them whole)")
    return Drawn(fig, data[[c for c in keep if c in data.columns]], notes)


# ============================================================================ figure 10


def fig10_closed_loop(inputs: Inputs, settings: dict) -> Drawn:
    """a: the weekly gain of refitting after the switch to closed loop in DCLP3 against its open-loop control; b:
    each arm against the model trained after the switch, week by week (72)."""
    weekly = points(inputs.csv("72_change_recovery", "weekly_gain"), ["gain", "low", "high"])
    after = points(inputs.csv("72_change_recovery", "trained_after"), ["difference", "low", "high"])
    arms = [("lightgbm_ft_noexog", "restart_every1", "-", "every day"),
            ("lstm_ft_noexog", "warm_every3", "-", "continued every 3 days")]

    fig = paper_figure(105)
    axes, strip = horizon_grid(fig, 2, sharey_rows=False)
    rows = []
    for column, horizon in enumerate(HORIZONS):
        ax = axes[0, column]
        for model, policy, _, text in arms:
            for group, style, alpha in (("dclp3_to_closed_loop", "-", 0.12), ("dclp3_sap_control", (0, (3, 1.5)),
                                                                               0.0)):
                part = weekly[(weekly["group"] == group) & (weekly["model"] == model) & (weekly["policy"] == policy)
                              & (weekly["horizon_min"] == horizon)].sort_values("week")
                if part.empty:
                    continue
                week = part["week"] + 1
                if alpha:
                    ax.fill_between(week, part["low"], part["high"], color=colour(model), alpha=alpha, linewidth=0)
                ax.plot(week, part["gain"], color=colour(model), linestyle=style, linewidth=0.9,
                        marker=marker(model), markersize=2.8)
                rows += [{"panel": "a", "horizon_min": horizon, "group": group, "model": model, "policy": policy,
                          "week": int(w), "gain_pts": g, "low": lo, "high": hi, "events": int(n)}
                         for w, g, lo, hi, n in zip(week, part["gain"], part["low"], part["high"], part["events"])]
        ax.axhline(0, **ZERO)
        ax.set_xticks(range(1, 7))
        if column == 1:
            ax.set_xlabel("Week after the switch (closed loop) or the same day (open-loop control)")
        if column == 0:
            ax.set_ylabel("Gain over frozen\n(points)")
            ax.legend(handles=[Line2D([], [], color=TEXT_SECONDARY, label="switch to Control-IQ"),
                               Line2D([], [], color=TEXT_SECONDARY, linestyle=(0, (3, 1.5)), label="SAP control")],
                      loc="upper left")

        ax = axes[1, column]
        for model, refit, _, text in arms:
            for policy, style in (("never", (0, (1, 1))), (refit, "-")):
                part = after[(after["model"] == model) & (after["policy"] == policy)
                             & (after["horizon_min"] == horizon)].sort_values("week")
                if part.empty:
                    continue
                ax.fill_between(part["week"], part["low"], part["high"], color=colour(model), alpha=0.10,
                                linewidth=0)
                ax.plot(part["week"], part["difference"], color=colour(model), linestyle=style, linewidth=0.9,
                        marker=marker(model), markersize=2.8)
                rows += [{"panel": "b", "horizon_min": horizon, "model": model, "policy": policy, "week": int(w),
                          "difference_pts": d, "low": lo, "high": hi}
                         for w, d, lo, hi in zip(part["week"], part["difference"], part["low"], part["high"])]
        ax.axhline(0, **ZERO)
        ax.set_xticks(range(1, 6))
        if column == 1:
            ax.set_xlabel("Week after the switch to closed loop")
        if column == 0:
            ax.set_ylabel("Minus the model trained\nafter the switch (points)")
            ax.legend(handles=[Line2D([], [], color=TEXT_SECONDARY, linestyle=(0, (1, 1)), label="frozen"),
                               Line2D([], [], color=TEXT_SECONDARY, label="refitted")], loc="lower right")
    for row in axes:
        same_limits(row)
    hide_inner_y(axes)
    handles = [Line2D([], [], color=colour(m), marker=marker(m), markersize=3, label=f"{label(m)}, {t}")
               for m, _, _, t in arms]
    bottom_legend(strip, handles)
    notes = ["DCLP3 (72): the glucose-only twins, which run on both arms; a: switch from SAP to Control-IQ against "
             "the SAP control; b: the model trained after the switch is the frozen model of the Control-IQ cohort "
             "(50) on the same days"]
    return Drawn(fig, pd.DataFrame(rows), notes)


# ============================================================================ figure 11


def fig11_lows(inputs: Inputs, settings: dict) -> Drawn:
    """a: the mean forecast for each reference glucose (shrinkage); b: alarm curves for a low (< 70 mg/dL), with the
    weighted loss (71); c: Clarke zones D and E (50). REPLACE-BG hold-out."""
    calibration = inputs.csv("70_low_glucose_analysis", "calibration")
    alarms = inputs.csv("71_low_glucose_loss", "alarm_curves")
    zones = inputs.csv("50_confirmation_runs", "clarke_zones")
    cohort = "replace_bg/core"
    curves = [("persistence", "reference", "standard", "Persistence", (0, (1, 1))),
              ("arx", "online_l1", "standard", "ARX, every sample", "-"),
              ("lightgbm_ft", "never", "standard", "LightGBM, frozen", (0, (3, 1.5))),
              ("lightgbm_ft", "restart_every3", "standard", "LightGBM, refitted", "-"),
              ("lightgbm_ft", "restart_every3", "x5", "LightGBM, refitted, lows x5", (0, (5, 1, 1, 1)))]
    fig = paper_figure(166)
    axes, strip = horizon_grid(fig, 3, heights=[1.1, 1.1, 0.9], sharey_rows=False, legend_rows=2)
    rows = []
    for column, horizon in enumerate(HORIZONS):
        ax = axes[0, column]
        frame = calibration[(calibration["cohort"] == cohort) & (calibration["horizon_min"] == horizon)
                            & (calibration["kind"] == "forecast_given_reference")]
        for model, policy, _, text, style in curves[:4]:
            part = frame[(frame["model"] == model) & (frame["policy"] == policy) & (frame["n"] >= 50)]
            part = part.sort_values("bin")
            ax.plot(part["bin"], part["mean"], color=colour(model), linestyle=style, linewidth=0.9, label=text)
            rows += [{"panel": "a", "horizon_min": horizon, "model": model, "policy": policy, "reference_mgdl": b,
                      "mean_forecast_mgdl": m, "n": int(n)} for b, m, n in zip(part["bin"], part["mean"], part["n"])]
        ax.plot([40, 400], [40, 400], color=TEXT_SECONDARY, linewidth=0.5)
        ax.axvspan(40, 70, color="#fbebeb", linewidth=0, zorder=0)
        ax.set_xlim(40, 400)
        ax.set_ylim(40, 400)
        if column == 1:
            ax.set_xlabel("Reference glucose (mg/dL)")
        if column == 0:
            ax.set_ylabel("Mean forecast (mg/dL)")

        ax = axes[1, column]
        frame = alarms[(alarms["cohort"] == cohort) & (alarms["horizon_min"] == horizon)
                       & (alarms["event"] == "below70")]
        for model, policy, variant, text, style in curves:
            part = frame[(frame["model"] == model) & (frame["policy"] == policy) & (frame["variant"] == variant)]
            part = part.sort_values("alarm")
            if part.empty:
                continue
            ax.plot(part["sensitivity_pct"], part["precision_pct"], color=colour(model), linestyle=style,
                    linewidth=0.9, label=text)
            at = part[part["alarm"] == 70]
            if not at.empty:
                ax.plot(at["sensitivity_pct"], at["precision_pct"], marker="o", color=colour(model),
                        markersize=3.2, linestyle="none")
            rows += [{"panel": "b", "horizon_min": horizon, "model": model, "policy": policy, "variant": variant,
                      "alarm_mgdl": a, "sensitivity_pct": s, "precision_pct": p}
                     for a, s, p in zip(part["alarm"], part["sensitivity_pct"], part["precision_pct"])]
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        if column == 1:
            ax.set_xlabel("Sensitivity for glucose < 70 mg/dL (%; alarm threshold 40 to 150 mg/dL)")
        if column == 0:
            ax.set_ylabel("Precision (%)")

        ax = axes[2, column]
        frame = zones[(zones["cohort"] == cohort) & (zones["horizon_min"] == horizon)]
        entries = [("persistence", "reference"), ("arx", "never"), ("arx", "online_l1"), ("lightgbm_ft", "never"),
                   ("lightgbm_ft", "restart_every3"), ("lstm_ft", "never"), ("lstm_ft", "warm_every3"),
                   ("tsmixerx_ft", "never"), ("tsmixerx_ft", "restart_other_every14"), ("tide_ft", "never"),
                   ("tide_ft", "restart_every14")]
        x = 0
        ticks, names = [], []
        for model, policy in entries:
            hit = frame[(frame["model"] == model) & (frame["policy"] == policy)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            frozen = policy in ("never", "reference")
            ax.bar(x, r["clarke_D_pct"], width=0.75, color=colour(model) if not frozen else "white",
                   edgecolor=colour(model), linewidth=0.6)
            ax.bar(x, r["clarke_E_pct"], bottom=r["clarke_D_pct"], width=0.75, color=TEXT_PRIMARY, linewidth=0)
            rows.append({"panel": "c", "horizon_min": horizon, "model": model, "policy": policy,
                         "clarke_D_pct": r["clarke_D_pct"], "clarke_E_pct": r["clarke_E_pct"],
                         "clarke_AB_pct": r["clarke_AB_pct"]})
            if policy != "never":
                ticks.append(x - (0 if policy == "reference" else 0.5))
                names.append(label(model))
            x += 1 if not frozen or policy == "reference" else 0.85
        ax.set_xticks(ticks)
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.grid(axis="x", visible=False)
        if column == 0:
            ax.set_ylabel("Clarke zone D + E (%)")
    hide_inner_y(axes[:2])
    handles = [Line2D([], [], color=colour(m), linestyle=s, linewidth=0.9, label=t) for m, _, _, t, s in curves]
    handles += [Line2D([], [], color=TEXT_SECONDARY, marker="o", linestyle="none", markersize=3.2,
                       label="b: alarm at 70 mg/dL")]
    handles += [Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=TEXT_SECONDARY, linewidth=0.6,
                          label="c: frozen"),
                Rectangle((0, 0), 1, 1, facecolor=TEXT_SECONDARY, linewidth=0, label="c: refitted"),
                Rectangle((0, 0), 1, 1, facecolor=TEXT_PRIMARY, linewidth=0, label="c: zone E")]
    bottom_legend(strip, handles, ncols=5)
    notes = ["REPLACE-BG hold-out, 105 patients; a: bins of 5 mg/dL with at least 50 forecasts (70); b: alarm when "
             "the forecast is below the threshold, dot at 70 mg/dL (71: lows weighted 5 times in every fit); c: "
             "zone E stacked in black (50)"]
    return Drawn(fig, pd.DataFrame(rows), notes)


# ============================================================================ the list

FIGURES = {
    "fig01_design": (fig01_design, "double", "a design; b protocol"),
    "fig02_population_personal": (fig02_population_personal, "double", "a ladder; b LightGBM personal only"),
    "fig03_families": (fig03_families, "double", "a frozen; b refitted; c skill and cost"),
    "fig04_ageing": (fig04_ageing, "double", "a against a model that cannot age; b slopes; c recency and data"),
    "fig05_policy": (fig05_policy, "double", "a window x interval; b share of the daily gain; c gain and cost"),
    "fig06_data": (fig06_data, "double", "a screening by week; b Loop by week; c later anchors"),
    "fig07_mechanism": (fig07_mechanism, "double", "a mechanisms; b context; c IS-LSTM"),
    "fig08_inputs": (fig08_inputs, "double", "a insulin and carbohydrates; b wearables"),
    "fig09_replication": (fig09_replication, "double", "gain at the optimal interval, screening and cohorts"),
    "fig10_closed_loop": (fig10_closed_loop, "double", "a gain after the switch; b against the model trained after"),
    "fig11_lows": (fig11_lows, "double", "a shrinkage; b alarm curves; c Clarke zones D and E"),
}


# ============================================================================ summary


def write_summary(table: pd.DataFrame, notes: list[str], font: str, results: Path, release, resources) -> str:
    drawn = table[table["drawn"]]
    lines = [
        "# 90 - Paper figures",
        "",
        f"Code release {release}. {resources.describe()}",
        f"Results read from `{results}`. Font: {font} (the list: {', '.join(PAPER_FONTS)}). "
        f"Text 7 pt (ticks and legends {PAPER_SMALL_PT:g} pt, never below {PAPER_MIN_PT:g} pt); "
        f"widths: {', '.join(f'{k} {v:g} mm' for k, v in PAPER_WIDTH_MM.items())}. No figure title: the conclusion "
        "goes in the caption. Each figure as PDF (vector, fonts embedded) and PNG (300 dpi), with its data.",
        "",
        f"## Figures ({len(drawn)} of {len(table)} drawn)",
        "",
        markdown_table(table[["figure", "drawn", "width_mm", "height_mm", "panels", "inputs"]]),
        "",
        "## Notes",
        "",
    ]
    lines += [f"- {note}" for note in notes] or ["- none"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
