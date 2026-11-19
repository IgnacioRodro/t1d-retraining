"""Paper figures (script 90): printed size and saving without changing it."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from PIL import Image  # noqa: E402

from t1dretrain import outputs  # noqa: E402
from t1dretrain.plotting import MM, PAPER_WIDTH_MM, paper_figure, panel_letter, set_paper_style  # noqa: E402


def test_paper_figures_keep_their_printed_size(tmp_path, monkeypatch):
    monkeypatch.setattr(outputs, "RESULTS_DIR", tmp_path)
    set_paper_style()
    fig = paper_figure(80)
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])
    panel_letter(ax, "a")
    ax.set_ylabel("A label long enough to need room on the left")
    assert abs(fig.get_size_inches()[0] / MM - PAPER_WIDTH_MM["double"]) < 1e-6
    out = outputs.ScriptOutput("90_test")
    png = out.figure(fig, "figure", dpi=300, formats=("png", "pdf"), exact_size=True)
    width, height = Image.open(png).size
    assert width == round(190 * MM * 300) and abs(height - 80 * MM * 300) < 1.5  # 2244 x 945 px: not cut
    assert (out.dir / "figure.pdf").exists()


def test_other_scripts_still_get_one_png(tmp_path, monkeypatch):
    monkeypatch.setattr(outputs, "RESULTS_DIR", tmp_path)
    fig = paper_figure(80)
    fig.subplots().plot([0, 1], [0, 1])
    out = outputs.ScriptOutput("90_test")
    png = out.figure(fig, "figure.png", dpi=100)
    assert png.name == "figure.png" and sorted(p.name for p in out.dir.iterdir()) == ["figure.png"]


def test_the_bundled_font_is_known_to_matplotlib():
    from matplotlib import font_manager

    from t1dretrain.plotting import FONT_DIR, register_bundled_fonts

    register_bundled_fonts()
    known = {entry.fname for entry in font_manager.fontManager.ttflist}
    bundled = sorted(FONT_DIR.glob("*.ttf"))
    assert bundled and all(str(path) in known for path in bundled)
    assert {entry.name for entry in font_manager.fontManager.ttflist if entry.fname in map(str, bundled)} == {
        "Liberation Sans"}
