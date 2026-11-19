"""Tables and figures of the stage-0 evaluations (scripts 10 and 12).

The input is the table of day metrics returned by ``sim.stage0`` (one row per
patient, period, model, seed, horizon and test day, with sums of errors and
Clarke zone counts). Every table reports RMSE and the Clarke zones (rule R6);
skill is 1 - RMSE / RMSE of persistence of the same patient, period and
horizon; intervals are bootstrap intervals over patients (2000 replicates).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..plotting import SEQUENTIAL_BLUE, TEXT_SECONDARY
from ..sim.stage0 import FAMILIES, STATISTICAL_MODELS, TWIN, is_learned, is_twin, split_name, twin_of
from .metrics import ZONES, plot_clarke
from .stats import bootstrap_mean

FAMILY_LABELS = {"lightgbm": "LightGBM", "nhits": "N-HiTS", "tide": "TiDE", "tsmixerx": "TSMixerx", "lstm": "LSTM",
                 "patchtst": "PatchTST"}
# light, medium and dark shade of one hue per family: scratch, population, fine-tuned
FAMILY_COLORS = {"lightgbm": ("#8fd9bb", "#1baf7a", "#0b6e4a"), "nhits": ("#bdb4f0", "#6250d6", "#3a2c99"),
                 "tide": ("#f6c48f", "#e8871e", "#9c5410"), "tsmixerx": ("#f2a7c3", "#d6457a", "#8f2350"),
                 "lstm": ("#a9d4f2", "#3593d6", "#1b5c8f"), "patchtst": ("#d9d18f", "#b3a21b", "#6e630b")}
MODEL_LABELS = {"persistence": "Persistence", "linear_extrapolation": "Linear extrapolation (30 min)", "ar3": "AR(3)",
                "arx": "ARX"}
MODEL_COLORS = {"persistence": "#9b9a96", "linear_extrapolation": "#eb6834", "ar3": "#2a78d6", "arx": "#5a6b7d"}
for _family, _label in FAMILY_LABELS.items():
    for _suffix, _text, _shade in (("", "", 0), ("_pop", " population", 1), ("_ft", " fine-tuned", 2)):
        MODEL_LABELS[_family + _suffix] = _label + _text
        MODEL_COLORS[_family + _suffix] = FAMILY_COLORS[_family][_shade]


def _greyed(hex_colour: str, share: float = 0.55) -> str:
    """The colour mixed with grey: the glucose-only twin of a model (D76, D90)."""
    rgb = [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round((1 - share) * c + share * 155):02x}" for c in rgb)


for _family, _label in FAMILY_LABELS.items():  # glucose-only twins (D76, D90): fine-tuned, glucose only
    MODEL_LABELS[f"{_family}_ft_{TWIN}"] = f"{_label} fine-tuned, glucose only"
    MODEL_COLORS[f"{_family}_ft_{TWIN}"] = _greyed(FAMILY_COLORS[_family][2])
MODEL_LABELS[f"arx_{TWIN}"], MODEL_COLORS[f"arx_{TWIN}"] = "ARX, glucose only", _greyed(MODEL_COLORS["arx"])
for _name, _label, _colour in (("chronos2", "Chronos-2", "#7a3fa0"), ("timesfm", "TimesFM 2.5", "#c43c6b"),
                               ("ttm", "TTM", "#b8860b")):  # foundation models, zero-shot (script 13)
    MODEL_LABELS[_name], MODEL_COLORS[_name] = _label, _colour
MODEL_LABELS["chronos2_univariate"] = "Chronos-2, glucose only"
MODEL_COLORS["chronos2_univariate"] = _greyed(MODEL_COLORS["chronos2"])
PERIOD_LABELS = {"post_randomization": "after randomisation", "whole": "whole follow-up"}


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def color(model: str) -> str:
    return MODEL_COLORS.get(model, TEXT_SECONDARY)


SUMS = ["n", "sse", "sae", "sape", "se"] + [f"zone_{z}" for z in ZONES]


def finish(frame: pd.DataFrame) -> pd.DataFrame:
    frame["rmse"] = np.sqrt(frame["sse"] / frame["n"])
    frame["mae"] = frame["sae"] / frame["n"]
    frame["mard_pct"] = 100 * frame["sape"] / frame["n"]
    frame["bias"] = frame["se"] / frame["n"]
    for z in ZONES:
        frame[f"clarke_{z}_pct"] = 100 * frame[f"zone_{z}"] / frame["n"]
    frame["clarke_AB_pct"] = frame["clarke_A_pct"] + frame["clarke_B_pct"]
    return frame.drop(columns=["sse", "sae", "sape", "se"] + [f"zone_{z}" for z in ZONES])


def patient_table(days: pd.DataFrame) -> pd.DataFrame:
    keys = ["period", "model", "seed", "horizon_min", "patient_id"]
    table = finish(days.groupby(keys, as_index=False)[SUMS].sum())
    reference = table[table["model"] == "persistence"].set_index(["period", "horizon_min", "patient_id"])["rmse"]
    table["skill"] = 1 - table["rmse"] / table.set_index(["period", "horizon_min", "patient_id"]).index.map(reference)
    return table.round(4)


def overall_table(days: pd.DataFrame, by_patient: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Pooled forecasts per period, model, seed and horizon (R6), with patient means and bootstrap intervals."""
    keys = ["period", "model", "seed", "horizon_min"]
    table = finish(days.groupby(keys, as_index=False)[SUMS].sum())
    rows = []
    for _, row in table.iterrows():
        p = by_patient[(by_patient[keys] == row[keys]).all(axis=1)]
        rmse = bootstrap_mean(p["rmse"].to_numpy(), replicates, seed)
        skill = bootstrap_mean(p["skill"].to_numpy(), replicates, seed)
        rows.append({"patients": int(len(p)), "patient_rmse_mean": rmse[0], "patient_rmse_low": rmse[1],
                     "patient_rmse_high": rmse[2], "skill_mean": skill[0], "skill_low": skill[1],
                     "skill_high": skill[2]})
    return pd.concat([table, pd.DataFrame(rows)], axis=1).round(4)


def model_summary(by_patient: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Per period, model and horizon: patient RMSE and skill averaged over seeds, bootstrap intervals, seed spread."""
    rows = []
    for (period, model, horizon), g in by_patient.groupby(["period", "model", "horizon_min"], sort=False):
        per_patient = g.groupby("patient_id")[["rmse", "skill", "clarke_AB_pct"]].mean()
        rmse = bootstrap_mean(per_patient["rmse"].to_numpy(), replicates, seed)
        skill = bootstrap_mean(per_patient["skill"].to_numpy(), replicates, seed)
        seed_means = g.groupby("seed")["skill"].mean()
        rows.append({"period": period, "model": model, "horizon_min": horizon, "seeds": int(g["seed"].nunique()),
                     "patients": int(len(per_patient)), "patient_rmse_mean": rmse[0], "patient_rmse_low": rmse[1],
                     "patient_rmse_high": rmse[2], "skill_mean": skill[0], "skill_low": skill[1], "skill_high": skill[2],
                     "skill_seed_sd": float(seed_means.std()) if len(seed_means) > 1 else np.nan,
                     "clarke_AB_pct_mean": float(per_patient["clarke_AB_pct"].mean())})
    table = pd.DataFrame(rows)
    order = {m: k for k, m in enumerate(MODEL_LABELS)}
    return table.sort_values(["period", "horizon_min", "model"], key=lambda c: c.map(order) if c.name == "model"
                             else c).round(4).reset_index(drop=True)


def noise_floor(by_patient: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """For each learned model: mean skill difference between every pair of seeds, with patients resampled.

    The DMD is the 95th percentile of the absolute bootstrap means of those differences: the change in mean skill
    that two trainings of the same model can show by chance (seed and patient sampling, D61). Each model draws its
    own bootstrap stream (from ``seed`` and its period, name and horizon), so a model's DMD does not depend on the
    other models of the table. ``dmd_stage`` (the largest DMD of the full models of the period and horizon) is
    kept for reference; comparisons use the DMD of each pair (D75).
    """
    rows = []
    learned = by_patient[by_patient["model"].map(is_learned)]
    for (period, model, horizon), g in learned.groupby(["period", "model", "horizon_min"]):
        wide = g.pivot_table(index="patient_id", columns="seed", values="skill").dropna()
        seeds = list(wide.columns)
        if len(seeds) < 2 or len(wide) < 3:
            continue
        rng = np.random.default_rng([int(seed), stable_hash(f"{period}|{model}|{int(horizon)}")])
        absolute, observed = [], []
        for i, a in enumerate(seeds):
            for b in seeds[i + 1:]:
                d = (wide[a] - wide[b]).to_numpy()
                observed.append(d.mean())
                means = d[rng.integers(0, d.size, size=(replicates, d.size))].mean(axis=1)
                absolute.append(np.abs(means))
        absolute = np.concatenate(absolute)
        rows.append({"period": period, "model": model, "horizon_min": horizon, "seeds": len(seeds),
                     "patients": int(len(wide)), "mean_abs_seed_difference": float(np.mean(np.abs(observed))),
                     "max_abs_seed_difference": float(np.max(np.abs(observed))),
                     "patient_seed_sd_median": float(wide.std(axis=1).median()),
                     "dmd": float(np.quantile(absolute, 0.95))})
    table = pd.DataFrame(rows)
    if not table.empty:
        full = table[~table["model"].map(is_twin)]
        stage = full.groupby(["period", "horizon_min"])["dmd"].max().rename("dmd_stage")
        table = table.join(stage, on=["period", "horizon_min"])
    return table.round(5)


def stable_hash(text: str) -> int:
    """A hash of ``text`` that is the same in every run and on every machine (Python's ``hash`` is not)."""
    import zlib

    return zlib.crc32(text.encode())


def difficulty_table(days: pd.DataFrame) -> pd.DataFrame:
    """Daily RMSE of persistence (how hard each day is) by period, horizon and week of follow-up."""
    p = days[days["model"] == "persistence"].copy()
    p["rmse"] = np.sqrt(p["sse"] / p["n"])
    p["week"] = (p["age_days"] // 7).astype(int)
    rows = []
    for (period, horizon, week), g in p.groupby(["period", "horizon_min", "week"]):
        rows.append({"period": period, "horizon_min": horizon, "week": week, "patient_days": len(g),
                     "rmse_mean": g["rmse"].mean(), "rmse_q25": g["rmse"].quantile(0.25),
                     "rmse_median": g["rmse"].median(), "rmse_q75": g["rmse"].quantile(0.75)})
    return pd.DataFrame(rows).round(3)


# ============================================================================ figures


def plot_rmse(summary: pd.DataFrame, periods: list[str], models: list[str], title: str) -> plt.Figure:
    horizons = sorted(summary["horizon_min"].unique())
    fig, axes = plt.subplots(1, len(periods), figsize=(7 * len(periods), 4.6), sharey=True, squeeze=False)
    width = 0.8 / len(models)
    for ax, period in zip(axes[0], periods):
        o = summary[summary["period"] == period]
        for k, model in enumerate(models):
            m = o[o["model"] == model].set_index("horizon_min").reindex(horizons)
            x = np.arange(len(horizons)) + (k - (len(models) - 1) / 2) * width
            err = [m["patient_rmse_mean"] - m["patient_rmse_low"], m["patient_rmse_high"] - m["patient_rmse_mean"]]
            ax.bar(x, m["patient_rmse_mean"], width=width, color=color(model),
                   yerr=err, capsize=2, label=label(model))
            for xx, v in zip(x, m["patient_rmse_mean"]):
                ax.text(xx, v + 1, f"{v:.0f}", ha="center", fontsize=6, color=TEXT_SECONDARY)
        ax.set_xticks(np.arange(len(horizons)), [f"{h} min" for h in horizons])
        ax.set_title(PERIOD_LABELS.get(period, period), loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("RMSE (mg/dL), mean over patients (95 % CI)")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(title, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_skill(by_patient: pd.DataFrame, period: str, models: list[str]) -> plt.Figure:
    b = by_patient[(by_patient["period"] == period) & (by_patient["model"] != "persistence")]
    b = b.groupby(["model", "horizon_min", "patient_id"], as_index=False)["skill"].mean()
    models = [m for m in models if m != "persistence" and m in set(b["model"])]
    horizons = sorted(b["horizon_min"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.8 * len(horizons), 4.2), sharey=True, squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        data = [b[(b["model"] == m) & (b["horizon_min"] == horizon)]["skill"].dropna() for m in models]
        parts = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True)
        for patch, model in zip(parts["boxes"], models):
            patch.set_facecolor(color(model))
            patch.set_alpha(0.6)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8, ls="--")
        ax.set_xticks(range(1, len(models) + 1), [label(m).split(" (")[0] for m in models], fontsize=8,
                      rotation=15)
        ax.set_title(f"{horizon} min", loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("skill = 1 - RMSE / RMSE of persistence (per patient)")
    fig.suptitle(f"Skill over persistence per patient, {PERIOD_LABELS.get(period, period)}", fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_noise(noise: pd.DataFrame, by_patient: pd.DataFrame, period: str) -> plt.Figure:
    n = noise[noise["period"] == period]
    models = [m for m in MODEL_LABELS if m in set(n["model"])] + sorted(set(n["model"]) - set(MODEL_LABELS))
    horizons = sorted(n["horizon_min"].unique())
    cols = min(3, len(models))
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.8 * rows), sharey=True, squeeze=False)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    for ax, model in zip(axes.flat, models):
        m = n[n["model"] == model].set_index("horizon_min").reindex(horizons)
        x = np.arange(len(horizons))
        ax.bar(x, m["dmd"], width=0.5, color=color(model), alpha=0.5, label="DMD (95 % of seed-pair differences)")
        seeds = by_patient[(by_patient["period"] == period) & (by_patient["model"] == model)]
        means = seeds.groupby(["horizon_min", "seed"])["skill"].mean().reset_index()
        for k, horizon in enumerate(horizons):
            values = means[means["horizon_min"] == horizon]["skill"].to_numpy()
            diffs = [abs(a - b) for i, a in enumerate(values) for b in values[i + 1:]]
            ax.scatter([k] * len(diffs), diffs, color=color(model), zorder=3, s=20,
                       label="observed seed-pair difference" if k == 0 else None)
        ax.set_xticks(x, [f"{h} min" for h in horizons])
        ax.set_title(label(model), loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("absolute difference in mean skill")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Noise floor: how much two trainings of the same model differ, {PERIOD_LABELS.get(period, period)}",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_grids(histogram: pd.DataFrame, models: list[str], horizons: list[int], period: str, seed: int,
               group: str = "screen") -> plt.Figure:
    fig, axes = plt.subplots(len(models), len(horizons), figsize=(4.3 * len(horizons), 4.5 * len(models)),
                             squeeze=False)
    for i, model in enumerate(models):
        model_seed = seed if is_learned(model) else 0
        for j, horizon in enumerate(horizons):
            h = histogram[(histogram["model"] == model) & (histogram["seed"] == model_seed) &
                          (histogram["horizon_min"] == horizon) & (histogram["period"] == period)]
            plot_clarke(axes[i, j], histogram=h, title=f"{label(model)}, {horizon} min")
            if j:
                axes[i, j].set_ylabel("")
            if i < len(models) - 1:
                axes[i, j].set_xlabel("")
    fig.suptitle(f"Clarke Error Grid, {group} group, {PERIOD_LABELS.get(period, period)} (density, log scale; "
                 f"learned models: seed {seed})", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_difficulty(difficulty: pd.DataFrame, periods: list[str]) -> plt.Figure:
    horizons = sorted(difficulty["horizon_min"].unique())
    fig, axes = plt.subplots(1, len(periods), figsize=(6.5 * len(periods), 4.2), sharey=True, squeeze=False)
    for ax, period in zip(axes[0], periods):
        d = difficulty[difficulty["period"] == period]
        for k, horizon in enumerate(horizons):
            h = d[(d["horizon_min"] == horizon) & (d["patient_days"] >= 10)]
            color = SEQUENTIAL_BLUE[2 + 2 * k]
            ax.plot(h["week"], h["rmse_median"], color=color, lw=2, label=f"{horizon} min")
            ax.fill_between(h["week"], h["rmse_q25"], h["rmse_q75"], color=color, alpha=0.15, linewidth=0)
        ax.set_xlabel("weeks after the 14 training days")
        ax.set_title(PERIOD_LABELS.get(period, period), loc="left", fontsize=10)
    axes[0, 0].set_ylabel("daily RMSE of persistence (mg/dL), median and IQR")
    axes[0, 0].legend(title="horizon")
    fig.suptitle("How hard the days are over the follow-up (persistence has no parameters)", fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ paired comparisons


def pairs_for(models: set) -> list[tuple[str, str]]:
    """(model, against) pairs: each starting point against the others of its family and against AR(3); every
    fine-tuned network against fine-tuned LightGBM (the best model of it. 18); ARX against AR(3); each glucose-only
    twin against its full model (D76, D90)."""
    pairs = []
    for f in FAMILIES:
        names = [n for n in (f, f"{f}_pop", f"{f}_ft") if n in models]
        for i, a in enumerate(names):
            for b in names[:i]:
                pairs.append((a, b))
        if "ar3" in models:
            pairs += [(n, "ar3") for n in names]
        if f != "lightgbm" and f"{f}_ft" in models and "lightgbm_ft" in models:
            pairs.append((f"{f}_ft", "lightgbm_ft"))
    if "arx" in models and "ar3" in models:
        pairs.append(("arx", "ar3"))
    for full in [f"{f}_ft" for f in FAMILIES] + list(STATISTICAL_MODELS):
        twin = twin_of(full, FAMILIES)
        if full in models and twin in models:
            pairs.append((full, twin))
    return pairs


def read_noise_floor(path) -> pd.DataFrame:
    """The ``noise_floor.csv`` of an earlier script; empty if it is missing or has no rows (too few patients)."""
    from pathlib import Path

    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


class DMDTable:
    """DMD of a model or of a pair of models (D75, replaces the stage-wide DMD of D61 and D67).

    A learned model has its own DMD (its seed noise, from a noise-floor table). A model without seed noise
    (reference, statistical and foundation models, or a learned model run with one seed) takes the floor: the
    smallest DMD among the learned models of the table for that period and horizon. Glucose-only twins (D90) keep
    their own DMD but do not set the floor, so that they change nothing for the full models. A comparison between
    two models uses the larger DMD of the two, so the noise of one model does not widen the ties of the others.
    ``models`` limits the table to some learned models (for example those of stage 1)."""

    def __init__(self, noise: pd.DataFrame | None, models: list[str] | None = None) -> None:
        noise = pd.DataFrame() if noise is None else noise
        if not noise.empty and models is not None:
            noise = noise[noise["model"].isin(models)]
        self.own = noise.set_index(["period", "model", "horizon_min"])["dmd"] if not noise.empty \
            else pd.Series(dtype=float)
        full = noise[~noise["model"].map(is_twin)] if not noise.empty else noise
        self.floor = full.groupby(["period", "horizon_min"])["dmd"].min() if not full.empty \
            else pd.Series(dtype=float)

    def __len__(self) -> int:
        return len(self.own)

    def model(self, period: str, horizon: int, model: str) -> float:
        value = float(self.own.get((period, model, horizon), np.nan))
        return value if np.isfinite(value) else float(self.floor.get((period, horizon), np.nan))

    def pair(self, period: str, horizon: int, a: str, b: str) -> float:
        values = [v for v in (self.model(period, horizon, a), self.model(period, horizon, b)) if np.isfinite(v)]
        return max(values) if values else np.nan


def paired_comparisons(by_patient: pd.DataFrame, noise: DMDTable | pd.DataFrame, replicates: int, seed: int,
                       pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Mean skill difference (seeds averaged per patient) for each (model, against) pair, with a bootstrap
    interval over patients. Verdict (D61, D75): ``tie`` if the difference is below the DMD of the pair (``noise``:
    a ``DMDTable`` or a noise-floor table), else ``better`` / ``worse`` when the interval excludes zero, else
    ``uncertain``."""
    skill = by_patient.groupby(["period", "model", "horizon_min", "patient_id"])["skill"].mean()
    dmd = noise if isinstance(noise, DMDTable) else DMDTable(noise)
    rows = []
    for (period, horizon), g in skill.groupby(level=["period", "horizon_min"]):
        wide = g.droplevel(["period", "horizon_min"]).unstack("model")
        for a, b in pairs:
            if a not in wide or b not in wide:
                continue
            d = (wide[a] - wide[b]).dropna().to_numpy()
            if len(d) < 3:
                continue
            mean, low, high = bootstrap_mean(d, replicates, seed)
            floor = dmd.pair(period, horizon, a, b)
            if np.isfinite(floor) and abs(mean) < floor:
                verdict = "tie"
            elif low > 0:
                verdict = "better"
            elif high < 0:
                verdict = "worse"
            else:
                verdict = "uncertain"
            rows.append({"period": period, "family": split_name(a)[0] or a, "horizon_min": horizon, "model": a,
                         "against": b, "patients": len(d), "skill_difference": mean, "low": low, "high": high,
                         "patients_better": int((d > 0).sum()), "dmd": floor, "verdict": verdict})
    return pd.DataFrame(rows).round(4)


def plot_paired(paired: pd.DataFrame, period: str) -> plt.Figure:
    p = paired[paired["period"] == period].copy()
    p["pair"] = p["model"].map(label) + " vs " + p["against"].map(label)
    horizons = sorted(p["horizon_min"].unique())
    pairs = list(dict.fromkeys(p["pair"]))
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.2 * len(horizons), 0.45 * len(pairs) + 1.8), sharey=True,
                             squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        h = p[p["horizon_min"] == horizon].set_index("pair").reindex(pairs)
        y = np.arange(len(pairs))[::-1]
        labelled = False
        for yy, (_, row) in zip(y, h.iterrows()):
            if pd.isna(row["skill_difference"]):
                continue
            c = color(row["model"])
            if np.isfinite(row["dmd"]):  # the tie band of this pair (D75)
                ax.fill_between([-row["dmd"], row["dmd"]], yy - 0.35, yy + 0.35, color=TEXT_SECONDARY, alpha=0.15,
                                lw=0, zorder=0, label=None if labelled else "DMD of the pair (tie)")
                labelled = True
            ax.plot([row["low"], row["high"]], [yy, yy], color=c, lw=2)
            ax.scatter(row["skill_difference"], yy, color=c, s=30, zorder=3,
                       marker="o" if row["verdict"] in ("better", "worse") else "s")
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_yticks(y, pairs, fontsize=8)
        ax.set_title(f"{horizon} min", loc="left", fontsize=10)
        ax.set_xlabel("difference in mean skill (95 % CI)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Paired comparisons of starting points, {PERIOD_LABELS.get(period, period)} (circle: better or "
                 "worse beyond the DMD; square: tie or uncertain)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- completeness of a run

DATA_ERRORS = ("ValueError: no training samples", "ValueError: too few training samples")
FIT_NOTE = ("`on_gpu`: fits done on a GPU. `cpu_fallback`: CUDA failed and the fit was done again on the CPU (valid, "
            "slower). `data_failure`: windows without usable samples (the same every run). `other_failure`: fits "
            "that failed for another reason (for example the GPU memory); the run is complete when it is 0.")


def fit_outcomes(fit_log: pd.DataFrame) -> pd.DataFrame:
    """Per model: fits, fits on a GPU, fits redone on the CPU after a CUDA failure, and failed fits by cause (D72)."""
    if fit_log.empty or "model" not in fit_log:
        return pd.DataFrame()
    log = fit_log.copy()
    error = log["error"] if "error" in log else pd.Series(None, index=log.index, dtype=object)
    cuda = log["cuda_error"].notna() if "cuda_error" in log else pd.Series(False, index=log.index)
    gpu = log["device"].astype(str).str.startswith("cuda") if "device" in log else pd.Series(False, index=log.index)
    failed = error.notna() & (error.astype(str) != "")
    data = failed & error.astype(str).str.startswith(DATA_ERRORS)
    table = pd.DataFrame({"model": log["model"], "fits": 1, "ok": (~failed).astype(int),
                          "on_gpu": (gpu & ~failed).astype(int), "cpu_fallback": (cuda & ~failed).astype(int),
                          "data_failure": data.astype(int), "other_failure": (failed & ~data).astype(int)})
    return table.groupby("model", as_index=False).sum()
