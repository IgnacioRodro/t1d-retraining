"""Shared figure style. All labels, titles and legends are written in English."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STATUS_COLORS = {
    "ok": "#2a9d8f",
    "wrong_version": "#e9c46a",
    "import_error": "#9d4edd",
    "missing": "#e76f51",
}


# One fixed colour per dataset in every figure of the project (categorical
# slots in a fixed order, validated for colour-vision deficiency on the adjacent
# pairs). Three of them are light on white, so figures always add direct labels,
# line styles or a table next to the colour.
DATASET_ORDER = ["replace_bg", "dclp3", "loop", "hupa_ucm", "brist1d_open", "t1d_uom", "diadata"]
DATASET_COLORS = {
    "replace_bg": "#2a78d6",
    "dclp3": "#eb6834",
    "loop": "#1baf7a",
    "hupa_ucm": "#eda100",
    "brist1d_open": "#e87ba4",
    "t1d_uom": "#008300",
    "diadata": "#6250d6",
}
DATASET_LABELS = {
    "replace_bg": "REPLACE-BG",
    "dclp3": "DCLP3",
    "loop": "Loop",
    "hupa_ucm": "HUPA-UCM",
    "brist1d_open": "BrisT1D",
    "t1d_uom": "T1D-UOM",
    "diadata": "DiaData",
}
DATASET_LINESTYLES = {
    "replace_bg": "-",
    "dclp3": "--",
    "loop": "-.",
    "hupa_ucm": ":",
    "brist1d_open": (0, (5, 1, 1, 1)),
    "t1d_uom": (0, (3, 1, 1, 1, 1, 1)),
    "diadata": (0, (1, 1)),
}
# Patient groups of the core dataset (script 04 and later)
GROUP_COLORS = {"pool": "#9b9a96", "screen": "#2a78d6", "confirm": "#eb6834"}
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def set_style() -> None:
    """Apply a clean, consistent style to every figure of the project."""
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 100,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
        }
    )


# ============================================================================ paper figures (script 90)
# Elsevier artwork sizes (single column, 1.5 column and full width; at most about 240 mm tall), 7 pt text and
# never below 6 pt; fonts embedded in the PDF. Arial if the machine has it, else Liberation Sans, which ships with
# the project (fonts/, SIL Open Font License 1.1) and has the same metrics: the figures are laid out the same on
# every machine (a server with only DejaVu Sans, wider, crowded the text of figure 1).
MM = 1 / 25.4  # inches per millimetre
PAPER_WIDTH_MM = {"single": 90.0, "one_and_half": 140.0, "double": 190.0}
PAPER_MAX_HEIGHT_MM = 240.0
PAPER_FONT_PT = 7.0
PAPER_SMALL_PT = 6.5
PAPER_MIN_PT = 6.0
PAPER_FONTS = ["Arial", "Liberation Sans", "DejaVu Sans"]
FONT_DIR = Path(__file__).resolve().parent / "fonts"
PAPER_GRID = "#e4e3df"
# One colour and one marker per model family in every paper figure. The colours are the categorical slots of the
# project in a fixed order, validated for colour-vision deficiency on adjacent pairs; only the first three pass on
# every pair, so a figure with more than three families also tells them apart by marker and label. The linear
# models and persistence are greys: references, not families.
PAPER_FAMILIES = {
    "lightgbm": ("LightGBM", "#2a78d6", "o"),
    "lstm": ("LSTM", "#eb6834", "s"),
    "tsmixerx": ("TSMixerx", "#1baf7a", "^"),
    "tide": ("TiDE", "#eda100", "D"),
    "nhits": ("N-HiTS", "#e87ba4", "v"),
    "patchtst": ("PatchTST", "#008300", "P"),
    "chronos2": ("Chronos-2", "#6250d6", "X"),
    "timesfm": ("TimesFM 2.5", "#e34948", "*"),
    "arx": ("ARX", "#52514e", "h"),
    "ar3": ("AR(3)", "#9b9a96", "p"),
    "persistence": ("Persistence", "#8a8984", "."),
}

# The name of every source of results in the paper's figures and tables (scripts 90 and 91).
PAPER_COHORTS = {
    "screen (stage 3)": "Screening", "screen": "Screening", "replace_bg/core": "REPLACE-BG",
    "dclp3/sap_open_insulin": "DCLP3 SAP", "dclp3/sap_open_glucose": "DCLP3 SAP, glucose only",
    "dclp3/clc_closed_insulin": "DCLP3 Control-IQ", "loop/closed_insulin": "Loop",
    "brist1d_open/open_insulin": "BrisT1D, open loop", "brist1d_open/closed_insulin": "BrisT1D, closed loop",
    "t1d_uom/open_insulin": "T1D-UOM", "t1d_uom/open_glucose": "T1D-UOM, glucose only",
    "hupa_ucm/open_insulin": "HUPA-UCM",
}


def register_bundled_fonts() -> None:
    """Make the fonts that ship with the project (``fonts/*.ttf``) known to matplotlib."""
    from matplotlib import font_manager

    known = {entry.fname for entry in font_manager.fontManager.ttflist}
    for path in sorted(FONT_DIR.glob("*.ttf")):
        if str(path) not in known:
            font_manager.fontManager.addfont(str(path))


def set_paper_style() -> None:
    """The style of the paper figures: 7 pt text, thin lines, light grid, no top or right spine, embedded fonts."""
    register_bundled_fonts()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": PAPER_FONTS,
            "font.size": PAPER_FONT_PT,
            "axes.labelsize": PAPER_FONT_PT,
            "axes.titlesize": PAPER_FONT_PT,
            "axes.titleweight": "normal",
            "axes.titlepad": 3,
            "axes.labelpad": 2,
            "axes.linewidth": 0.5,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PAPER_GRID,
            "grid.linewidth": 0.4,
            "grid.alpha": 1.0,
            "xtick.labelsize": PAPER_SMALL_PT,
            "ytick.labelsize": PAPER_SMALL_PT,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelcolor": TEXT_PRIMARY,
            "ytick.labelcolor": TEXT_PRIMARY,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.5,
            "lines.markeredgewidth": 0.6,
            "errorbar.capsize": 0,
            "legend.fontsize": PAPER_SMALL_PT,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 1.0,
            "legend.borderaxespad": 0.2,
            "text.color": TEXT_PRIMARY,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.dpi": 100,
            "figure.constrained_layout.h_pad": 0.03,
            "figure.constrained_layout.w_pad": 0.03,
            "figure.constrained_layout.hspace": 0.04,
            "figure.constrained_layout.wspace": 0.03,
        }
    )


def paper_figure(height_mm: float, width: str = "double") -> plt.Figure:
    """An empty figure of the final printed size (no scaling at print), laid out by constrained layout."""
    return plt.figure(figsize=(PAPER_WIDTH_MM[width] * MM, height_mm * MM), layout="constrained")


def panel_letter(ax, letter: str) -> None:
    """The bold panel letter at the top left of the panel (the figure itself has no title: the caption says it)."""
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=PAPER_FONT_PT + 1)


def model_axes(n: int, width: float, height: float, ncols: int = 3):
    """Panels of one figure, one per model, at most ``ncols`` per row (unused panels hidden): (figure, panels)."""
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols + 0.6, height * nrows + 0.6), squeeze=False)
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]
