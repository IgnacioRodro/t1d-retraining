"""41 - Stage 3 analysis: cost and efficiency of each way of taking in new data, insulin and carbohydrates, forgetting.

Reads the day metrics and the fit log of script 40 (HANDOFF 7.5, D86;
``eval.mechanism_analysis``) and answers Q3 on the core dataset:

- **arms**: per model, the skill of every arm (mean over patients, per
  horizon), its gain over never, its computing seconds per test day and its
  efficiency (gain over never, in skill points, per computing second per day);
  every arm against the reference (restart from the population model, D65;
  ARX: the update at every sample without forgetting, D87) with the tie rule of D61 and
  the model's DMD (D75), in skill points and mg/dL;
- **over the weeks**: the daily difference of each arm with the reference,
  week by week since the anchor (whether a warm start drifts away);
- **forgetting** (ARX): the skill against the forgetting factor and its
  effective memory (the leaf refits of LightGBM with decay 0.9, 0.5 and 0, D88,
  are arms of the tables above);
- **insulin and carbohydrates** (D90): for every model and its glucose-only
  twin, the gain of each arm over never with every input and with the glucose
  only, and their paired difference;
- **mechanism for the confirmation** (D91): per model, the arm with the largest
  gain over never among those that beat the reference at one horizon at least
  and lose to it at none, otherwise the reference; written to
  ``mechanisms.yaml`` for the confirmation (script 50). Twins take the arm of
  their full model.

Every evaluation reports RMSE and the Clarke zones (rule R6) per model, arm and
horizon; the Clarke grid data are in script 40.

Inputs:
    <derived_data>/runs/40_stage3_mechanism_runs/day_metrics.parquet and fit_log.parquet
    results/12_stage0_population_transfer/noise_floor.csv   (DMD, D75)

Outputs (results/41_stage3_mechanism_analysis/):
    arm_table.csv            per model and arm: skill per horizon, gain over never, seconds per day, efficiency
    vs_reference.csv         every arm against the reference arm of its model (paired, verdict)
    weekly_difference.csv    daily difference with the reference by week since the anchor (patient means, 95 %)
    forgetting.csv           ARX: skill and difference with the reference (no forgetting) per factor and horizon
    exog_mechanism.csv       D90: gain over never of every arm with every input and with the glucose only
    mechanisms.yaml          D91: the arm of each model for the confirmation (also in <derived_data>/runs/41_...)
    metrics_by_arm.csv       per model, arm and horizon: n, RMSE, MAE, MARD, bias, Clarke zones (R6)
    efficiency.png, arms_by_week.png, forgetting.png, exog_mechanism.png
    summary.md, run_manifest.json

Usage:
    python scripts/41_stage3_mechanism_analysis.py
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
import yaml  # noqa: E402

from t1dretrain.compute import detect_resources  # noqa: E402
from t1dretrain.config import CONFIG_DIR, derived_data_dir, load_project_config  # noqa: E402
from t1dretrain.eval import mechanism_analysis as ma  # noqa: E402
from t1dretrain.eval import policy_analysis as pa  # noqa: E402
from t1dretrain.eval import policy_report as pr  # noqa: E402
from t1dretrain.eval.stage0_report import (PERIOD_LABELS, SUMS, DMDTable, color, finish, label,  # noqa: E402
                                           read_noise_floor)
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import SEQUENTIAL_BLUE, TEXT_SECONDARY, model_axes, set_style  # noqa: E402
from t1dretrain.sim.mechanisms import parse_arm  # noqa: E402
from t1dretrain.sim.stage0 import FAMILIES, split_name, split_variant  # noqa: E402

SCRIPT_ID = "41_stage3_mechanism_analysis"
SOURCE = "40_stage3_mechanism_runs"
HORIZON_COLORS = {30: SEQUENTIAL_BLUE[2], 60: SEQUENTIAL_BLUE[4], 120: SEQUENTIAL_BLUE[6]}
ARM_COLORS = {"never": "#9b9a96", "restart": "#2a78d6", "restart_all": "#1baf7a", "restart_head": "#1baf7a",
              "restart_other": "#1baf7a",
              "warm": "#d1495b", "leaves": "#eb6834", "scratch": "#7b4fd6", "online": "#3d5a80"}


def mix(hex_color: str, other: str, share: float) -> str:
    """``hex_color`` moved ``share`` of the way towards ``other`` (RGB)."""
    a = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    b = np.array([int(other[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    c = np.clip(np.round(a + share * (b - a)), 0, 255).astype(int)
    return "#" + "".join(f"{v:02x}" for v in c)


def arm_palette(arms) -> dict:
    """A colour per arm: the colour of its kind, in shades from light to dark when a kind has several values
    (ARX forgetting factors, LightGBM leaf decays), ordered by the value."""
    kinds: dict = {}
    for arm in arms:
        kind, value = parse_arm(arm)
        kinds.setdefault(kind, []).append((value if value is not None else 0.0, arm))
    out = {}
    for kind, items in kinds.items():
        base = ARM_COLORS.get(kind, "#5a6b7d")
        items.sort()
        n = len(items)
        for i, (_, arm) in enumerate(items):
            share = 0.0 if n == 1 else 0.6 - 0.95 * i / (n - 1)  # light (towards white) to dark (towards black)
            out[arm] = mix(base, "#ffffff", share) if share >= 0 else mix(base, "#000000", -share)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


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
    stage3, cfg = config["stage3"], config.get("stage2_analysis", {})
    replicates, seed = int(config["stage0"]["bootstrap_replicates"]), int(config["random_seed"])
    min_origins, clip = int(cfg.get("min_origins_per_day", 12)), tuple(cfg.get("skill_clip", [-1.0, 1.0]))
    test_days = int(config["anchors"]["max_age_days"])
    period = list(stage3["periods"])[0]
    runs = derived / "runs" / SOURCE
    if not (runs / "day_metrics.parquet").exists():
        raise SystemExit(f"{runs / 'day_metrics.parquet'} not found: run scripts/40_stage3_mechanism_runs.py first")
    days = pd.read_parquet(runs / "day_metrics.parquet")
    days = days[days["period"] == period]
    fit_log = pd.read_parquet(runs / "fit_log.parquet") if (runs / "fit_log.parquet").exists() else pd.DataFrame()
    used = fit_log[fit_log["excluded"].isna()] if "excluded" in fit_log else fit_log
    try:
        manifest = json.loads((Path(out.dir).parent / SOURCE / "run_manifest.json").read_text())["settings"]
    except Exception:  # noqa: BLE001
        manifest = {}
    present = [m for m in days["model"].unique() if m != "persistence"]
    models = [m for m in manifest.get("models", []) if m in present] + \
        [m for m in present if m not in manifest.get("models", [])]
    references = manifest.get("references") or {m: "restart" for m in models}
    arms_of = manifest.get("arms") or {m: sorted(days.loc[days["model"] == m, "policy"].unique()) for m in models}
    dmd = DMDTable(read_noise_floor(Path(out.dir).parent / "12_stage0_population_transfer" / "noise_floor.csv"),
                   [m for m in models if split_name(m)[0] in FAMILIES])
    print(f"{len(days)} day rows; models {models}")

    skill = pr.patient_skill(days)
    table = ma.arm_table(skill, used, test_days, references)
    vs_reference = pd.concat([pr.paired(skill[skill["model"] == m], lambda a, ref=references.get(m):
                                        ref if a not in (ref, "never") else None, dmd, replicates, seed)
                              for m in models], ignore_index=True)
    vs_never = pr.paired(skill, lambda a: "never", dmd, replicates, seed)
    weekly = weekly_difference(days, models, references, min_origins, clip)
    forgetting = ma.forgetting_table(skill, vs_reference)
    exog = pa.exog_effect(skill, dmd, replicates, seed)
    metrics = finish(days.groupby(["period", "model", "policy", "horizon_min"], as_index=False)[SUMS].sum()).round(4)
    chosen = ma.confirmation_mechanisms(vs_reference, table, models, references, period)
    document = {"release": release, "period": period, "intervals": manifest.get("intervals"),
                "rule": "the arm with the largest gain over never among those that beat the reference at one horizon "
                        "at least and lose to it at none (tie rule with the model's DMD); otherwise the reference; "
                        "a glucose-only twin takes the arm of its full model (D91)",
                "models": chosen}
    (derived / "runs" / SCRIPT_ID).mkdir(parents=True, exist_ok=True)
    (derived / "runs" / SCRIPT_ID / "mechanisms.yaml").write_text(yaml.safe_dump(document, sort_keys=False))

    out.csv(table.round(5), "arm_table")
    out.csv(vs_reference.round(4), "vs_reference")
    out.csv(vs_never.round(4), "vs_never")
    out.csv(weekly.round(4), "weekly_difference")
    out.csv(forgetting.round(5), "forgetting")
    out.csv(exog.round(4), "exog_mechanism")
    out.csv(metrics, "metrics_by_arm")
    out.text(yaml.safe_dump(document, sort_keys=False), "mechanisms.yaml")
    base_models = [m for m in models if not split_variant(m)[1]]
    if not table.empty:
        out.figure(plot_efficiency(table, vs_never, base_models, period), "efficiency")
    if not weekly.empty:
        out.figure(plot_weekly(weekly, base_models, references, period), "arms_by_week")
    if not forgetting.empty and (forgetting.groupby("model")["forgetting"].nunique() > 1).any():
        out.figure(plot_forgetting(forgetting, period), "forgetting")
    if not exog.empty:
        out.figure(plot_exog(exog, period), "exog_mechanism")
    out.text(write_summary(table, vs_reference, forgetting, exog, weekly, chosen, models, arms_of, references,
                           manifest, release, resources, period, len(days)), "summary.md")
    out.manifest(settings={"period": period, "models": models, "references": references,
                           "intervals": manifest.get("intervals"), "bootstrap_replicates": replicates},
                 notes="Inputs: day metrics and fit log of script 40; DMD from script 12.")
    print(f"Done. Results in {out.dir}")


def weekly_difference(days: pd.DataFrame, models: list[str], references: dict, min_origins: int,
                      clip: tuple) -> pd.DataFrame:
    """Daily skill of every arm minus that of its model's reference arm, by week since the anchor."""
    parts = []
    for model in models:
        d = days[days["model"].isin([model, "persistence"])]
        arms = [a for a in d.loc[d["model"] == model, "policy"].unique() if a not in (references.get(model), "never")]
        gain = pa.daily_gain(d, arms, references.get(model), min_origins, clip)
        if not gain.empty:
            parts.append(pa.weekly_gain(gain))
    return pd.concat(parts, ignore_index=True).rename(columns={"gain_mean": "difference", "gain_low": "low",
                                                              "gain_high": "high"}) if parts else pd.DataFrame()


# ============================================================================ figures


def plot_efficiency(table: pd.DataFrame, vs_never: pd.DataFrame, models: list[str], period: str):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    t = table[(table["period"] == period) & (table["seconds_per_day"] > 0)]
    fig, axes = model_axes(len(models), 4.2, 3.5)
    for ax, model in zip(axes, models):
        m = t[t["model"] == model].sort_values("gain_over_never")
        palette = arm_palette(m["arm"])
        span = max(1e-9, 100 * (m["gain_over_never"].max() - m["gain_over_never"].min()))
        last, k = None, 0
        for _, r in m.iterrows():
            x, y = r["seconds_per_day"], 100 * r["gain_over_never"]
            ax.scatter(x, y, s=34, zorder=3, color=palette.get(r["arm"], color(model)), edgecolors="white", lw=0.6)
            crowded = last is not None and y - last[1] < 0.07 * span and abs(np.log10(x / last[0])) < 0.35
            k = k + 1 if crowded else 0  # neighbours with almost the same gain and cost: alternate sides, then stack
            left = k % 2 == 1
            ax.annotate(r["arm"], (x, y), fontsize=6, textcoords="offset points",
                        xytext=(-6 if left else 6, 3 + 9 * (k // 2)), ha="right" if left else "left",
                        arrowprops={"arrowstyle": "-", "lw": 0.4, "color": TEXT_SECONDARY} if k >= 2 else None)
            last = (x, y)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_xscale("log")
        low, high = m["seconds_per_day"].min(), m["seconds_per_day"].max()
        narrow = len(m) and high / max(low, 1e-12) < 30  # under about a decade and a half: label 1, 2 and 5
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0) if narrow else (1.0,)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:g} s"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.margins(x=0.25, y=0.15)
        ax.set_title(label(model), loc="left", fontsize=9)
        ax.set_xlabel("computing time per test day")
    if models:
        axes[0].set_ylabel("gain over never, mean of horizons (points)")
    fig.suptitle(f"Gain against cost of each way of taking in new data (D86), {PERIOD_LABELS.get(period, period)}",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_weekly(weekly: pd.DataFrame, models: list[str], references: dict, period: str, horizon: int = 60):
    w = weekly[(weekly["period"] == period) & (weekly["horizon_min"] == horizon)]
    shown = [m for m in models if m in set(w["model"])]
    fig, axes = model_axes(max(1, len(shown)), 4.2, 3.2)
    for ax, model in zip(axes, shown):
        m = w[w["model"] == model]
        palette = arm_palette(m["policy"].unique())
        for arm, a in m.groupby("policy"):
            a = a.sort_values("week")
            c = palette.get(arm, color(model))
            ax.plot(a["week"], 100 * a["difference"], color=c, lw=1.5, marker="o", ms=2.5, label=arm)
            ax.fill_between(a["week"], 100 * a["low"], 100 * a["high"], color=c, alpha=0.10, lw=0)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_title(f"{label(model)} (against {references.get(model)})", loc="left", fontsize=9)
        ax.set_xlabel("weeks since the anchor")
        ax.legend(fontsize=6)
    if shown:
        axes[0].set_ylabel(f"daily difference at {horizon} min (points)")
    fig.suptitle(f"Each arm against the reference, week by week, {PERIOD_LABELS.get(period, period)}",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_forgetting(forgetting: pd.DataFrame, period: str):
    f = forgetting[forgetting["period"] == period]
    models = [m for m, g in f.groupby("model") if g["forgetting"].nunique() > 1]  # a single factor has no curve
    fig, axes = model_axes(len(models), 4.6, 3.6)
    for ax, model in zip(axes, models):
        m = f[f["model"] == model]
        finite = m[np.isfinite(m["memory_days"])]
        top = finite["memory_days"].max() * 3 if not finite.empty else 30.0  # λ = 1 drawn at the right
        for horizon, h in m.groupby("horizon_min"):
            h = h.sort_values("forgetting")
            x = h["memory_days"].where(np.isfinite(h["memory_days"]), top)
            ax.plot(x, 100 * h["skill_mean"], marker="o", ms=4, lw=1.5,
                    color=HORIZON_COLORS.get(int(horizon), TEXT_SECONDARY), label=f"{int(horizon)} min")
        ax.set_xscale("log")
        ticks = sorted(set(finite["memory_days"].round(3))) + [top]
        ax.set_xticks(ticks, [f"{t:.2g}" for t in ticks[:-1]] + ["no\nforgetting"], fontsize=7)
        ax.minorticks_off()
        ax.set_title(f"{label(model)}: memory of the update at every sample", loc="left", fontsize=9)
        ax.set_xlabel("effective memory 1 / (1 - λ) (days)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("skill (%)")
    fig.suptitle(f"ARX forgetting factor, {PERIOD_LABELS.get(period, period)}", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_exog(exog: pd.DataFrame, period: str):
    e = exog[exog["period"] == period]
    pairs = sorted(set(zip(e["model"], e["twin"])))
    fig, axes = model_axes(len(pairs), 4.4, 3.4)
    for ax, (base, twin) in zip(axes, pairs):
        m = e[(e["model"] == base) & (e["twin"] == twin)]
        arms = sorted(m["policy"].unique())
        horizons = sorted(m["horizon_min"].unique())
        width = 0.8 / max(1, len(horizons))
        for j, horizon in enumerate(horizons):
            h = m[m["horizon_min"] == horizon].set_index("policy")
            x = np.arange(len(arms)) + (j - (len(horizons) - 1) / 2) * width
            c = HORIZON_COLORS.get(int(horizon), TEXT_SECONDARY)
            ax.bar(x, [100 * h.loc[a, "gain_all_inputs"] for a in arms], width * 0.9, color=c,
                   label=f"{int(horizon)} min, every input")
            ax.scatter(x, [100 * h.loc[a, "gain_glucose_only"] for a in arms], s=18, marker="D", color="white",
                       edgecolors=c, zorder=3, label=f"{int(horizon)} min, glucose only")
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_xticks(np.arange(len(arms)), arms, fontsize=7)
        ax.set_title(label(base), loc="left", fontsize=9)
    axes[0].set_ylabel("gain over never (points)")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"Gain of each arm with and without insulin and carbohydrates (bars: every input; diamonds: glucose "
                 f"only), {PERIOD_LABELS.get(period, period)}", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ============================================================================ summary


def write_summary(table, vs_reference, forgetting, exog, weekly, chosen, models, arms_of, references, manifest,
                  release, resources, period, n_rows) -> str:
    lines = ["# 41 - Stage 3 analysis: how each model takes in new data", "", f"Code release {release}.",
             f"Machine: {resources.describe()}",
             f"{n_rows} day rows of script 40; models and arms {arms_of}; refit intervals "
             f"{manifest.get('intervals')}; references {references}; {PERIOD_LABELS.get(period, period)}.", ""]
    if not table.empty:
        t = table[table["period"] == period].copy()
        for c in ("gain_over_never", "gain_vs_reference"):
            t[c] = (100 * t[c]).round(2)
        show = [c for c in ("model", "arm", "skill_30", "skill_60", "skill_120", "gain_over_never",
                            "gain_vs_reference", "seconds_per_day", "cost_vs_reference", "efficiency", "patients")
                if c in t.columns]
        lines += ["## Arms (skill; gain in points; efficiency: points of gain over never per computing second per "
                  "day)", "", markdown_table(t[show].round(4)), ""]
    if not vs_reference.empty:
        v = vs_reference[vs_reference["period"] == period].copy()
        v["cell"] = [f"{100 * d:+.2f} ({100 * lo:+.2f} to {100 * hi:+.2f}; {verdict})" for d, lo, hi, verdict in
                     zip(v["skill_difference"], v["low"], v["high"], v["verdict"])]
        wide = v.pivot_table(index=["model", "policy", "against"], columns="horizon_min", values="cell",
                             aggfunc="first")
        wide.columns = [f"{int(c)} min" for c in wide.columns]
        lines += ["## Every arm against the reference (skill points, 95 % CI, verdict with the model's DMD)", "",
                  markdown_table(wide.reset_index()), ""]
    if not forgetting.empty:
        f = forgetting[forgetting["best"]]
        lines += ["## ARX: best forgetting factor per horizon", "",
                  markdown_table(f[["model", "horizon_min", "policy", "forgetting", "memory_days",
                                    "skill_mean"]].round(4)), "",
                  "Every factor, with its difference from the reference arm, is in `forgetting.csv`.", ""]
    if not exog.empty:
        e = exog[exog["period"] == period].copy()
        for col in ("gain_all_inputs", "gain_glucose_only", "difference", "low", "high", "dmd"):
            e[col] = (100 * e[col]).round(2)
        lines += ["## Insulin and carbohydrates (D90): gain over never of every arm with every input and with the "
                  "glucose only (points)", "",
                  markdown_table(e[["model", "policy", "horizon_min", "patients", "gain_all_inputs",
                                    "gain_glucose_only", "difference", "low", "high", "verdict"]]), ""]
    if not weekly.empty:
        w = weekly[(weekly["period"] == period) & (weekly["horizon_min"] == 60)]
        last = w[w["week"] >= w["week"].max() - 1].groupby(["model", "policy"])["difference"].mean()
        first = w[w["week"] <= 1].groupby(["model", "policy"])["difference"].mean()
        drift = pd.DataFrame({"weeks_0_1": 100 * first, "last_2_weeks": 100 * last}).round(2).reset_index()
        lines += ["## Over the weeks (60 min): difference with the reference in the first and the last two weeks "
                  "(points)", "", markdown_table(drift), ""]
    if chosen:
        rows = [{"model": m, "arm": c["arm"], "reference": c["reference"], "gain_over_never": c["gain_over_never_points"],
                 "against_reference": ", ".join(f"{h} min {x}" for h, x in c["against_reference"].items()),
                 "twin": c.get("twin", ""), "reason": c["reason"]} for m, c in chosen.items()]
        lines += ["## Mechanism for the confirmation (D91; `mechanisms.yaml`)", "",
                  "Gain over never in skill points (mean over horizons); the twin runs the arm of its full model.", "",
                  markdown_table(pd.DataFrame(rows)), ""]
    lines += ["## Checks to look at", "",
              "- `efficiency.png`: which way of taking in new data gives the most gain for its cost.",
              "- `arms_by_week.png`: whether a warm start drifts away from restarting as the weeks go by.",
              "- `forgetting.png`: the memory that suits ARX best.",
              "- `exog_mechanism.png`: whether insulin and carbohydrates change which mechanism works.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
