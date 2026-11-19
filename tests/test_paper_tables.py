"""Paper tables (script 91): the plan of the confirmation, the share of the daily gain kept and the LaTeX cells."""

from __future__ import annotations

import pandas as pd
import pytest

from t1dretrain.eval import paper_tables as pt


def replication_rows() -> pd.DataFrame:
    rows = []
    for horizon, gain, loss in ((30, 0.0122, -0.0009), (60, 0.0186, -0.0013), (120, 0.0235, -0.0014)):
        rows += [
            {"cohort": "replace_bg/core", "model": "lightgbm_ft", "horizon_min": horizon, "finding": "gain_over_never",
             "policy": "restart_every3", "against": "never", "skill_difference": gain, "patients": 105},
            {"cohort": "replace_bg/core", "model": "lightgbm_ft", "horizon_min": horizon, "finding": "loss_to_daily",
             "policy": "restart_every3", "against": "restart_every1", "skill_difference": loss, "patients": 105},
            {"cohort": "replace_bg/core", "model": "lstm_ft", "horizon_min": horizon, "finding": "gain_over_never",
             "policy": "restart_every3", "against": "never", "skill_difference": 0.01, "patients": 105},
            {"cohort": "replace_bg/core", "model": "lstm_ft", "horizon_min": horizon,
             "finding": "mechanism_vs_restart", "policy": "warm_every3", "against": "restart_every3",
             "skill_difference": 0.004, "patients": 105},
            {"cohort": "replace_bg/core", "model": "lstm_ft_noexog", "horizon_min": horizon,
             "finding": "mechanism_vs_restart", "policy": "restart_other_every14", "against": "restart_every14",
             "skill_difference": 0.0, "patients": 105},
        ]
    return pd.DataFrame(rows)


def test_the_plan_of_the_confirmation_is_read_from_its_arms():
    arms = pt.recommended_arms(replication_rows()).set_index("model")
    assert arms.loc["lightgbm_ft", "arm"] == "restart" and arms.loc["lightgbm_ft", "interval_days"] == 3
    assert arms.loc["lstm_ft", "arm"] == "warm"
    assert "lstm_ft_noexog" not in arms.index  # twins take their full model's plan


def test_share_of_the_daily_gain_kept_is_the_ratio_of_the_published_gains():
    kept = pt.kept_share(replication_rows()).set_index(["model", "horizon_min"])["share_pct"]
    assert kept[("lightgbm_ft", 30)] == pytest.approx(100 * 0.0122 / (0.0122 + 0.0009))
    assert ("lstm_ft", 30) not in kept.index  # no daily refit to compare with


def test_cells_and_latex():
    assert pt.span(93.4, 94.3) == "93--94" and pt.span(80.2, 79.8) == "80"
    assert pt.triplet([22.24, 19.15, 19.5]) == "22.2/19.1/19.5"
    assert pt.count(32, 33) == "32/33" and pt.count(0, 0) == pt.NOT_APPLICABLE
    assert pt.latex_escape("Days (%) & more_x") == r"Days (\%) \& more\_x"
    assert pt.mechanism_text("tsmixerx_ft", "restart_other", "head") == "Refit all layers"
    assert "lambda=1" in pt.mechanism_text("arx", "online_l1", None)
    body = pt.latex_body(pd.DataFrame({"a": ["x"], "b": [r"\pend"]}), [("a", "A", "l"), ("b", "B", "X")])
    assert body.splitlines()[0].startswith(r"\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}X@{}}")
    assert r"x & \pend\\" in body


def test_replication_table_counts_per_cohort():
    summary = pd.DataFrame([
        {"cohort": "replace_bg/core", "finding": "gain_over_never", "comparisons": 33, "replicated": 32,
         "full_models": "18 of 18"},
        {"cohort": "replace_bg/core", "finding": "loss_to_daily", "comparisons": 33, "replicated": 33,
         "full_models": "18 of 18"},
        {"cohort": "loop/closed_insulin", "finding": "gain_over_never", "comparisons": 33, "replicated": 32,
         "full_models": "18 of 18"}])
    replication = pd.DataFrame({"cohort": ["replace_bg/core", "loop/closed_insulin"], "patients": [105, 120]})
    table = pt.replication_table(summary, replication, ["replace_bg/core", "loop/closed_insulin"], {})
    cells = pt.replication_cells(table)
    assert cells["gain"].tolist() == ["32/33", "32/33"] and cells["full"].tolist() == ["18/18", "18/18"]
    assert cells["daily"].tolist() == ["33/33", pt.NOT_APPLICABLE]


def test_models_table_reads_the_starting_points_the_pool_chose():
    chosen = {"lightgbm": {"population": {"num_leaves": 63, "learning_rate": 0.1},
                           "finetune": {"learning_rate": 0.01}},
              "lstm": {"population": {"layers": 2, "mlp_units": 256}, "finetune": {"train": "head"}},
              "nhits": {"population": {"mlp_units": 512}, "finetune": {"train": "all"}}}
    rows = [{"family": "ar3", "label": "AR(3)", "kind": "Linear", "inputs": "G", "start": "Personal, least squares",
             "refits": "Batch refit", "stages": "0–3, C"},
            {"family": "lightgbm", "label": "LightGBM", "kind": "Trees", "inputs": "G, I, C, T", "refits": "Restart",
             "stages": "0–3, C"},
            {"family": "lstm", "label": "LSTM", "kind": "Recurrent", "inputs": "G, I, C, T", "refits": "Restart",
             "stages": "0–3, C"},
            {"family": "nhits", "label": "N-HiTS", "kind": "MLP", "inputs": "G, I, C, T", "refits": "--",
             "stages": "0–1"},
            {"family": "tide", "label": "TiDE", "kind": "MLP", "inputs": "G, I, C, T", "refits": "--", "stages": "0–1"}]
    table = pt.models_table(rows, chosen)
    start = dict(zip(table["family"], table["start"]))
    assert start["ar3"] == "Personal, least squares"
    assert start["lightgbm"] == "Population (63 leaves, learning rate 0.1); fine-tuned by adding trees (learning rate 0.01)"
    assert start["lstm"] == "Population (2 layers of 256 units); fine-tuned on the output layers"
    assert start["nhits"] == "Population (512 units); fine-tuned on all layers"
    assert start["tide"] == pt.PENDING            # no population model in script 11
    cells = pt.models_cells(table)
    assert cells.loc[4, "start"] == pt.PENDING and cells.loc[0, "label"] == "AR(3)"
    body = pt.latex_body(cells, [("label", "Model", "l"), ("start", "Starting point", "X")], width=r"\textwidth")
    assert r"\begin{tabularx}{\textwidth}" in body and "fine-tuned on the output layers" in body
