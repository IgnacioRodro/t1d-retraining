"""70 - Forecast lows: why the learned models miss them (HANDOFF 7.10, D95).

The confirmation (script 51) found that the learned models detect few lows:
below 70 mg/dL at 30 min, a sensitivity of 25-32 % against 47 % for
persistence, with a higher precision. This script asks why, from what every
evaluation already saved (no model is run; ``eval.low_glucose``):

- **where the errors are**: bias, MAE, RMSE and Clarke zones per range of the
  reference glucose; zone D in the low range is a missed low;
- **shrinkage**: the mean forecast when the reference is low, and the mean
  reference when the forecast is low; E[forecast | reference] and E[reference
  | forecast] per 5-mg/dL bin (a model trained on the squared error forecasts
  the conditional mean, which regresses towards the middle when the future is
  uncertain);
- **alarm curves**: sensitivity and precision of an alarm that sounds when the
  forecast is below a threshold from 40 to 150 mg/dL, for the lows below 70
  (and 55, the bin edge next to 54) mg/dL; the sensitivity of every model at the
  precision of persistence (and the threshold it takes), and the average
  precision. A model with a higher curve detects lows better; one on the same
  curve only alarms more or less often;
- **per patient** (day metrics of script 50): sensitivity and precision below
  70 and 54 mg/dL with the alarm at the threshold, and paired differences
  against persistence, of the refits against never and of every model against
  its glucose-only twin (bootstrap over patients).

Two sources: the confirmation (script 50: every cohort, model and arm) and, if
present, stage 0 on the screen group (scripts 12 and 13), which adds the
foundation models used zero-shot (pending question of it. 21: why their
Clarke A+B is higher with a worse RMSE). Rule R6: every table has the RMSE and
the Clarke zones; the grid data are those of scripts 50, 12 and 13.

Inputs:
    results/50_confirmation_runs/clarke_histogram.csv and run_manifest.json
    <derived_data>/runs/50_confirmation_runs/day_metrics.parquet
    results/12_stage0_population_transfer/clarke_histogram.csv, results/13_stage0_foundation/clarke_histogram.csv

Outputs (results/70_low_glucose_analysis/):
    range_metrics.csv     per source, key and range of the reference: n, mean reference and forecast, bias, MAE,
                          RMSE (bin centres) and Clarke zones (R6)
    shrinkage.csv         mean forecast in the reference lows; mean reference of the forecast lows (below 70)
    calibration.csv       E[forecast | reference] and E[reference | forecast] per 5-mg/dL bin
    alarm_curves.csv      sensitivity, precision, F1 and false alarms for every alarm threshold
    matched_alarms.csv    sensitivity at the precision of persistence, precision at its sensitivity, average precision
    patient_lows.csv      per cohort, patient, model, arm, horizon and threshold: lows, alarms, hits, sensitivity
    paired_lows.csv       paired per patient: against persistence, refit against never, full model against twin
    zone_d_by_range.png, calibration.png, alarm_curves.png, patient_lows.png
    summary.md, run_manifest.json

Usage:
    python scripts/70_low_glucose_analysis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain.envguard import ensure_project_env  # noqa: E402  (standard library only)

ensure_project_env(__file__)  # re-runs this script inside the project environment if needed

from t1dretrain.codecheck import verify_code  # noqa: E402  (standard library only)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from t1dretrain.compute import detect_resources  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.eval import low_glucose as lg  # noqa: E402
from t1dretrain.eval.stage0_report import color, label  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, model_axes, set_style  # noqa: E402
from t1dretrain.sim.stage0 import CONTEXT_MODELS, FAMILIES, is_twin, split_name, split_variant  # noqa: E402

SCRIPT_ID = "70_low_glucose_analysis"
SOURCE = "50_confirmation_runs"
STAGE0 = ("12_stage0_population_transfer", "13_stage0_foundation")
CONFIRMATION_KEYS = ["cohort", "model", "policy", "horizon_min"]
STAGE0_KEYS = ["period", "model", "horizon_min"]
HISTOGRAM_TYPES = {"ref_bin": "float32", "pred_bin": "float32", "count": "int64", "horizon_min": "int16"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


def read_histogram(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=HISTOGRAM_TYPES)
    for column in ("cohort", "model", "policy", "period", "zone"):
        if column in table:
            table[column] = table[column].astype("category")
    return table


def stage0_histogram(results: Path) -> pd.DataFrame:
    """Stage 0 on the screen group: the models of script 12 and the foundation models of script 13 (pooled over
    seeds; persistence and the reference models from script 12 only, which scores the same origins)."""
    parts = []
    for script in STAGE0:
        path = results / script / "clarke_histogram.csv"
        if not path.exists():
            continue
        h = read_histogram(path)
        if script.startswith("13"):
            h = h[h["model"].isin(CONTEXT_MODELS)]
        parts.append(h)
    if not parts:
        return pd.DataFrame()
    h = pd.concat([p.astype({c: "object" for c in ("model", "period", "zone") if c in p}) for p in parts],
                  ignore_index=True)
    return lg.pooled(h, STAGE0_KEYS)


# ============================================================================ main


def main() -> None:
    import json

    args = parse_args()
    release = verify_code(strict=not args.skip_code_check)
    config = load_project_config()
    resources = detect_resources(config.get("compute", {}).get("reserve_cores"))
    print(resources.describe())
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    set_style()
    derived = derived_data_dir(args.paths)
    results = Path(out.dir).parent
    cfg = config.get("low_glucose", {})
    min_events, figure_cohort = int(cfg.get("min_events", 20)), str(cfg.get("figure_cohort", "replace_bg/core"))
    replicates, seed = int(config["stage0"]["bootstrap_replicates"]), int(config["random_seed"])
    path = results / SOURCE / "clarke_histogram.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found: run scripts/50_confirmation_runs.py first")
    manifest = json.loads((results / SOURCE / "run_manifest.json").read_text())
    plan = manifest["settings"]["plan"]
    confirmation = lg.pooled(read_histogram(path).astype({"cohort": "object", "model": "object", "policy": "object",
                                                          "zone": "object"}), CONFIRMATION_KEYS)
    stage0 = stage0_histogram(results)
    print(f"Confirmation histogram: {len(confirmation)} cells; stage 0: {len(stage0)} cells")

    tables = {name: [] for name in ("ranges", "shrinkage", "calibration", "curves", "matched")}
    for source, h, keys, ignore in (("confirmation", confirmation, CONFIRMATION_KEYS, ("policy",)),
                                    ("stage0", stage0, STAGE0_KEYS, ())):
        if h.empty:
            continue
        curves = lg.alarm_curves(h, keys)
        tables["ranges"].append(lg.range_table(h, keys).assign(source=source))
        tables["shrinkage"].append(lg.shrinkage(h, keys).assign(source=source))
        tables["calibration"].append(lg.calibration(h, keys).assign(source=source))
        tables["curves"].append(curves.assign(source=source))
        tables["matched"].append(lg.matched(curves, keys, ignore=ignore).assign(source=source))
    tables = {k: first(pd.concat(v, ignore_index=True)) if v else pd.DataFrame() for k, v in tables.items()}

    days_path = derived / "runs" / SOURCE / "day_metrics.parquet"
    patients, paired = pd.DataFrame(), pd.DataFrame()
    if days_path.exists():
        days = read_days(days_path)
        patients = lg.patient_lows(days, CONFIRMATION_KEYS, min_events)
        paired = paired_comparisons(patients, replicates, seed)
    else:
        print(f"{days_path} not found: no per-patient tables")

    out.csv(tables["ranges"].round(4), "range_metrics")
    out.csv(tables["shrinkage"].round(3), "shrinkage")
    out.csv(tables["calibration"].round(3), "calibration")
    out.csv(tables["curves"].round(3), "alarm_curves")
    out.csv(tables["matched"].round(3), "matched_alarms")
    out.csv(patients.round(3), "patient_lows")
    out.csv(paired.round(3), "paired_lows")
    views = figure_views(tables, figure_cohort)
    if views:
        out.figure(plot_zone_d(tables["ranges"], views), "zone_d_by_range")
        out.figure(plot_calibration(tables["calibration"], views), "calibration")
        out.figure(plot_alarm_curves(tables["curves"], views), "alarm_curves")
    if not patients.empty:
        out.figure(plot_patients(patients, figure_cohort, plan), "patient_lows")
    out.text(write_summary(tables, views, patients, paired, figure_cohort, min_events, release, resources, manifest),
             "summary.md")
    out.manifest(settings={"min_events": min_events, "figure_cohort": figure_cohort, "ranges": lg.RANGES,
                           "events": lg.EVENTS, "alarms": lg.ALARMS, "bootstrap_replicates": replicates,
                           "source_release": manifest.get("package_version")},
                 notes="Inputs: Clarke histograms of scripts 50, 12 and 13; day metrics of script 50.")
    print(f"Done. Results in {out.dir}")


def first(table: pd.DataFrame) -> pd.DataFrame:
    return table[["source"] + [c for c in table.columns if c != "source"]] if "source" in table else table


def read_days(path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema.names
    columns = CONFIRMATION_KEYS + ["patient_id"] + [c for c in names if c.startswith("below")]
    return pd.read_parquet(path, columns=columns)


def paired_comparisons(patients: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Per patient: every model against persistence (same cohort, horizon and threshold), every refit against
    never, every full model against its glucose-only twin."""
    if patients.empty:
        return pd.DataFrame()
    parts = []
    reference = patients[patients["model"] == lg.REFERENCE_MODEL]
    others = patients[patients["model"] != lg.REFERENCE_MODEL]
    copies = [reference.assign(policy=p) for p in others["policy"].unique()]  # persistence next to every arm
    table = pd.concat([others] + copies, ignore_index=True)
    models = sorted(others["model"].unique())
    parts.append(lg.paired_lows(table, CONFIRMATION_KEYS, "model", {m: lg.REFERENCE_MODEL for m in models},
                                replicates, seed).assign(comparison="model - persistence"))
    arms = sorted(a for a in others["policy"].unique() if a != "never")
    parts.append(lg.paired_lows(others, CONFIRMATION_KEYS, "policy", {a: "never" for a in arms}, replicates, seed)
                 .assign(comparison="refit - never"))
    twins = {split_variant(m)[0]: m for m in models if is_twin(m)}
    parts.append(lg.paired_lows(others, CONFIRMATION_KEYS, "model", twins, replicates, seed)
                 .assign(comparison="every input - glucose only"))
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    table = pd.concat(parts, ignore_index=True)
    return table[["comparison"] + [c for c in table.columns if c != "comparison"]]


# ============================================================================ figures


def interval_arm(model: str, plan: dict) -> str | None:
    k = (plan.get(model) or plan.get(split_variant(model)[0]) or {}).get("interval")
    return f"restart_every{int(k)}" if k else None


def figure_views(tables: dict, cohort: str) -> list[dict]:
    """The panels' rows: the figure cohort of the confirmation (never and persistence; full models) and stage 0
    after randomisation (fine-tuned learned models, reference models and foundation models)."""
    views = []
    curves = tables["curves"]
    if curves.empty:
        return views
    c = curves[(curves["source"] == "confirmation") & (curves["cohort"] == cohort)]
    if not c.empty:
        models = [m for m in dict.fromkeys(c["model"]) if not is_twin(m)]
        views.append({"name": f"confirmation, {cohort}, never", "source": "confirmation",
                      "filter": {"cohort": cohort, "policy": ["never", "reference"]}, "models": order(models)})
    s = curves[curves["source"] == "stage0"]
    if not s.empty:
        period = "post_randomization" if "post_randomization" in set(s["period"]) else sorted(set(s["period"]))[0]
        models = [m for m in dict.fromkeys(s["model"]) if not is_twin(m) and m != "chronos2_univariate" and
                  (split_name(m)[0] not in FAMILIES or split_name(m)[1] == "finetune")]
        views.append({"name": f"stage 0, screen group, {period.replace('_', ' ')}", "source": "stage0",
                      "filter": {"period": period}, "models": order(models)})
    return views


def order(models: list[str]) -> list[str]:
    """Persistence and the statistical models first, then the learned models, then the foundation models."""
    rank = {"persistence": 0, "ar3": 1, "arx": 2}
    return sorted(models, key=lambda m: (rank.get(m, 3 if split_name(m)[0] in FAMILIES else 4), m))


def view_rows(table: pd.DataFrame, view: dict) -> pd.DataFrame:
    t = table[table["source"] == view["source"]]
    for column, value in view["filter"].items():
        t = t[t[column].isin(value)] if isinstance(value, list) else t[t[column] == value]
    return t[t["model"].isin(view["models"])]


def plot_zone_d(ranges: pd.DataFrame, views: list[dict], horizons=(30, 60)):
    """Share of all forecasts in Clarke zone D, split by the range of the reference (low: a missed low)."""
    fig, axes = model_axes(len(views) * len(horizons), 5.4, 3.2, ncols=len(horizons))
    for i, view in enumerate(views):
        t = view_rows(ranges, view)
        for j, horizon in enumerate(horizons):
            ax = axes[i * len(horizons) + j]
            h = t[t["horizon_min"] == horizon]
            for k, model in enumerate(view["models"]):
                m = h[h["model"] == model]
                low = m.loc[m["range"].isin(["below 55", "55-69"]), "clarke_D_of_all_pct"].sum()
                high = m.loc[m["range"].isin(["180-249", "250 and above"]), "clarke_D_of_all_pct"].sum()
                ax.barh(k, low, color=color(model), height=0.7)
                ax.barh(k, high, left=low, color=color(model), alpha=0.35, height=0.7)
            ax.set_yticks(range(len(view["models"])), [label(m) for m in view["models"]], fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("zone D, % of all forecasts (dark: reference below 70; light: 180 and above)", fontsize=7)
            ax.set_title(f"{view['name']}, {horizon} min", loc="left", fontsize=8)
    fig.suptitle("Clarke zone D by range of the reference: the learned models' zone D is missed lows",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_calibration(calibration: pd.DataFrame, views: list[dict], horizons=(30, 60)):
    """Mean forecast for every reference bin (40-180 mg/dL): above the diagonal in the lows, a forecast too high."""
    fig, axes = model_axes(len(views) * len(horizons), 4.4, 3.6, ncols=len(horizons))
    for i, view in enumerate(views):
        t = view_rows(calibration, view)
        t = t[(t["kind"] == "forecast_given_reference") & (t["bin"] <= 180)]
        for j, horizon in enumerate(horizons):
            ax = axes[i * len(horizons) + j]
            h = t[t["horizon_min"] == horizon]
            for model in view["models"]:
                m = h[h["model"] == model].sort_values("bin")
                ax.plot(m["bin"], m["mean"], color=color(model), lw=1.3 if model != "persistence" else 2.0,
                        ls="--" if model == "persistence" else "-", label=label(model))
            ax.plot([40, 180], [40, 180], color=TEXT_SECONDARY, lw=0.7, ls=":")
            ax.axvline(70, color=TEXT_SECONDARY, lw=0.6)
            ax.axhline(70, color=TEXT_SECONDARY, lw=0.6)
            ax.set_xlim(40, 180)
            ax.set_ylim(40, 180)
            ax.set_xlabel("reference glucose (mg/dL)")
            ax.set_ylabel("mean forecast (mg/dL)")
            ax.set_title(f"{view['name']}, {horizon} min", loc="left", fontsize=8)
        axes[i * len(horizons)].legend(fontsize=6, loc="upper left")
    fig.suptitle("Mean forecast given the reference (5-mg/dL bins): above the diagonal in the lows is the shrinkage "
                 "towards the middle", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_alarm_curves(curves: pd.DataFrame, views: list[dict], horizons=(30, 60)):
    """Precision against sensitivity of an alarm at every forecast threshold (40-150 mg/dL; lows below 70)."""
    fig, axes = model_axes(len(views) * len(horizons), 4.4, 3.6, ncols=len(horizons))
    for i, view in enumerate(views):
        t = view_rows(curves, view)
        t = t[t["event"] == "below70"]
        for j, horizon in enumerate(horizons):
            ax = axes[i * len(horizons) + j]
            h = t[t["horizon_min"] == horizon]
            for model in view["models"]:
                m = h[h["model"] == model].sort_values("alarm")
                ax.plot(m["sensitivity_pct"], m["precision_pct"], color=color(model), lw=1.2,
                        ls="--" if model == "persistence" else "-", label=label(model))
                for alarm, marker in ((70, "o"), (80, "s"), (90, "^")):
                    point = m[m["alarm"] == alarm]
                    ax.scatter(point["sensitivity_pct"], point["precision_pct"], color=color(model), marker=marker,
                               s=14, zorder=3)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xlabel("sensitivity (%)")
            ax.set_ylabel("precision (%)")
            ax.set_title(f"{view['name']}, {horizon} min", loc="left", fontsize=8)
        axes[i * len(horizons)].legend(fontsize=6, loc="upper right")
    fig.suptitle("Alarm when the forecast is below a threshold, for a reference below 70 mg/dL (circle: 70, square: "
                 "80, triangle: 90 mg/dL)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_patients(patients: pd.DataFrame, cohort: str, plan: dict, horizons=(30, 60)):
    """Per patient of the figure cohort: sensitivity below 70 mg/dL of never and of the refit at the optimal
    interval (median and quartiles; dots: patients)."""
    t = patients[(patients["cohort"] == cohort) & (patients["threshold"] == 70)]
    models = order([m for m in dict.fromkeys(t["model"]) if not is_twin(m)])
    fig, axes = model_axes(len(horizons), 5.2, 0.42 * len(models) + 1.2, ncols=len(horizons))
    rng = np.random.default_rng(0)
    for ax, horizon in zip(axes, horizons):
        h = t[t["horizon_min"] == horizon]
        for k, model in enumerate(models):
            arms = ["reference"] if model == lg.REFERENCE_MODEL else ["never", interval_arm(model, plan)]
            for j, arm in enumerate(a for a in arms if a):
                v = h[(h["model"] == model) & (h["policy"] == arm)]["sensitivity_pct"].dropna()
                if v.empty:
                    continue
                y = k + (j - 0.5) * 0.36 if len(arms) > 1 else k
                q1, q2, q3 = np.quantile(v, [0.25, 0.5, 0.75])
                ax.scatter(v, y + rng.uniform(-0.08, 0.08, len(v)), s=3, color=color(model), alpha=0.35, lw=0)
                ax.plot([q1, q3], [y, y], color=color(model), lw=2.4, alpha=0.5 if arm == "never" else 0.95)
                ax.scatter([q2], [y], color=color(model), s=22, marker="|" if arm == "never" else "D", zorder=3)
        ax.set_yticks(range(len(models)), [label(m) for m in models], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("sensitivity below 70 mg/dL, per patient (%)")
        ax.set_title(f"{cohort}, {horizon} min (upper: never; lower: refit at the optimal interval)", loc="left",
                     fontsize=8)
    fig.suptitle("Forecast lows per patient (alarm when the forecast is below 70 mg/dL)", fontweight="bold", x=0.01,
                 ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(tables, views, patients, paired, cohort, min_events, release, resources, manifest) -> str:
    lines = ["# 70 - Forecast lows", "", f"Code release {release}.", f"Machine: {resources.describe()}",
             f"Clarke histograms of the confirmation (script 50, release {manifest.get('package_version')}) and of "
             "stage 0 (scripts 12 and 13), 5-mg/dL bins; per patient, the day metrics of script 50 (patients with "
             f"at least {min_events} lows, or alarms, for a sensitivity, or a precision). D95.", ""]
    for view in views:
        s = view_rows(tables["shrinkage"], view)
        m = view_rows(tables["matched"], view)
        m = m[m["event"] == "below70"]
        s = s[s["horizon_min"].isin([30, 60])].sort_values(["horizon_min", "model"])
        m = m[m["horizon_min"].isin([30, 60])].sort_values(["horizon_min", "model"])
        lines += [f"## {view['name']}: shrinkage of the forecasts (mg/dL)", "",
                  "`mean_forecast_in_lows`: mean forecast when the reference is below 70; "
                  "`mean_reference_of_forecast_lows`: mean reference when the forecast is below 70.", "",
                  markdown_table(s[["model", "horizon_min", "reference_lows", "mean_reference_in_lows",
                                    "mean_forecast_in_lows", "forecast_above_true_low",
                                    "mean_reference_of_forecast_lows"]].round(1)), "",
                  f"## {view['name']}: alarms for a reference below 70 mg/dL (%)", "",
                  "With the alarm at 70 mg/dL; and the most sensitive alarm that is at least as precise as "
                  "persistence's at 70 mg/dL: its sensitivity and threshold (`beyond`: still that precise at 150 "
                  "mg/dL; `below`: never that precise).", "",
                  markdown_table(m[["model", "horizon_min", "sensitivity_pct", "precision_pct",
                                    "sensitivity_at_reference_precision_pct", "alarm_for_reference_precision",
                                    "note_precision", "average_precision_pct"]].round(1)), ""]
    ranges = tables["ranges"]
    if not ranges.empty:
        r = ranges[(ranges["source"] == "confirmation") & (ranges["cohort"] == cohort) &
                   ranges["policy"].isin(["never", "reference"]) & (ranges["horizon_min"] == 30)]
        if not r.empty:
            r = r[~r["model"].map(is_twin)]
            lines += [f"## Confirmation, {cohort}, never, 30 min: errors by range of the reference", "",
                      markdown_table(r[["model", "range", "n", "share_pct", "bias", "rmse", "clarke_A_pct",
                                        "clarke_B_pct", "clarke_D_pct", "clarke_E_pct"]].round(2)), "",
                      "Every cohort, arm and horizon is in `range_metrics.csv`.", ""]
    if not paired.empty:
        p = paired[(paired["cohort"] == cohort) & (paired["threshold"] == 70) & (paired["measure"] == "sensitivity")]
        p = p[(p["comparison"] != "model - persistence") | (p["policy"] == "never")]
        if not p.empty:
            p = p.copy()
            p["difference (95 %)"] = [f"{d:+.1f} ({lo:+.1f} to {hi:+.1f})" for d, lo, hi in
                                      zip(p["difference"], p["low"], p["high"])]
            lines += [f"## Per patient, {cohort}: sensitivity below 70 mg/dL, paired differences (points)", "",
                      markdown_table(p[["comparison", "model", "policy", "against", "horizon_min", "patients",
                                        "mean_against", "difference (95 %)", "patients_higher"]]), "",
                      "Every cohort, threshold and the precision are in `paired_lows.csv`.", ""]
    lines += ["## Checks to look at", "",
              "- `alarm_curves.png`: whether the learned models' curves lie above persistence's (better detection) "
              "or on it (a more conservative alarm).",
              "- `calibration.png`: how far above the diagonal the mean forecast lies in the lows.",
              "- `zone_d_by_range.png`: whether zone D is missed lows or missed highs; the foundation models in "
              "stage 0.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
