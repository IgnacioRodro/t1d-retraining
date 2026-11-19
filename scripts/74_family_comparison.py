"""74 - Which model family forecasts best, frozen and refitted? Every family against every other (HANDOFF 7.10, D103).

The other analyses compare each model with itself (refitted against frozen,
one mechanism against another). Here the full models meet each other on the
same patients and days, in two states:

- **frozen**: never refitted after the fit at the anchor (``never``);
- **refitted**: as the study recommends for that model, its mechanism at its
  optimal interval (``eval.confirmation.recommended_arm``: D84, D91; ARX
  updated at every sample).

In a cohort without insulin, where the full models do not run, their
glucose-only twins (D90) stand in for them (column ``inputs``).

Sources: the screening (stage 3, script 40: the screen group from day 14, 84
test days; LightGBM, LSTM, TiDE, TSMixerx and ARX) and every cohort of the
confirmation (script 50: the same arms, AR(3) too). Paired per patient on the
patients both models have: the mean skill difference with a bootstrap interval;
tie below the DMD of the pair (D75; each learned model's from script 12, the
floor for the models without seed noise), better or worse when the interval
excludes zero, else uncertain. Per source, state and horizon, the model with
the highest mean skill on the patients every model has and the models that do
not lose to it. Nothing is chosen here: it describes the families. Rule R6:
the pooled RMSE and Clarke zones of these arms are in scripts 40 and 50.

Inputs:
    <derived_data>/runs/40_stage3_mechanism_runs/day_metrics.parquet; <derived_data>/runs/41_.../mechanisms.yaml
    <derived_data>/runs/50_confirmation_runs/day_metrics.parquet; results/50_confirmation_runs/run_manifest.json
    results/12_stage0_population_transfer/noise_floor.csv   (DMD, D75)

Outputs (results/74_family_comparison/):
    skill.csv            per source, state, model and horizon: patients, mean skill (95 % bootstrap) and RMSE
    pairwise.csv         every pair of models per source, state and horizon: difference (95 %), patients better,
                         DMD and verdict
    leaders.csv          per source, state and horizon: the best model on the common patients and those that do
                         not lose to it
    family_comparison.png  every model against the reference model (``family_comparison.reference``), per state
                         and horizon, in the screening and the large cohorts
    summary.md, run_manifest.json

Usage:
    python scripts/74_family_comparison.py
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
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
from t1dretrain.eval import confirmation as cf  # noqa: E402
from t1dretrain.eval import policy_report as pr  # noqa: E402
from t1dretrain.eval.stage0_report import DMDTable, color, label, read_noise_floor  # noqa: E402
from t1dretrain.eval.stats import bootstrap_mean  # noqa: E402
from t1dretrain.outputs import ScriptOutput, markdown_table  # noqa: E402
from t1dretrain.plotting import TEXT_SECONDARY, model_axes, set_style  # noqa: E402
from t1dretrain.sim.stage0 import FAMILIES, TWIN, is_twin, split_name  # noqa: E402

SCRIPT_ID = "74_family_comparison"
SCREEN = "screen (stage 3)"
STATES = ("frozen", "refitted")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="paths file (config/paths.yaml)")
    parser.add_argument("--skip-code-check", action="store_true",
                        help="run even if the code differs from the released manifest")
    return parser.parse_args()


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
    cfg = config.get("family_comparison", {})
    wanted = [str(m) for m in cfg.get("models", [])]
    reference = str(cfg.get("reference", "lightgbm_ft"))
    min_patients = int(cfg.get("min_patients", 20))
    replicates, seed = int(config["stage0"]["bootstrap_replicates"]), int(config["random_seed"])

    skills, notes = [], []
    screen = screen_skill(derived, wanted, notes)
    if not screen.empty:
        skills.append(screen.assign(source=SCREEN))
    confirm, cohorts = confirmation_skill(derived, results, wanted, notes)
    if not confirm.empty:
        skills.append(confirm)
    if not skills:
        raise SystemExit("no day metrics of scripts 40 or 50: run them first")
    skill = pd.concat(skills, ignore_index=True)
    models = [m for m in wanted if m in set(skill["entry"])]
    sources = [SCREEN] * (not screen.empty) + [c for c in cohorts if c in set(skill["source"])]
    dmd = DMDTable(read_noise_floor(results / "12_stage0_population_transfer" / "noise_floor.csv"),
                   sorted({m for m in skill["model"] if split_name(m)[0] in FAMILIES}))
    print(f"Sources {sources}; models {models}")

    means = mean_table(skill, replicates, seed)
    pairs = pairwise(skill, dmd, replicates, seed)
    leaders = leader_table(skill, pairs)

    out.csv(means.round(5), "skill")
    out.csv(pairs.round(5), "pairwise")
    out.csv(leaders, "leaders")
    sizes = skill.groupby("source")["patient_id"].nunique()
    shown = [s for s in sources if s == SCREEN or sizes.get(s, 0) >= min_patients]
    if not pairs.empty and reference in models:
        out.figure(plot_reference(pairs, reference, models, shown), "family_comparison")
    out.text(write_summary(means, pairs, leaders, sources, shown, models, reference, notes, release, resources),
             "summary.md")
    out.manifest(settings={"models": models, "reference": reference, "sources": sources, "shown": shown,
                           "bootstrap_replicates": replicates, "min_patients": min_patients},
                 notes="Day metrics of scripts 40 (screen) and 50 (confirmation); DMD from 12. "
                       + json.dumps({"notes": notes}))
    print(f"Done. Results in {out.dir}")


# ============================================================================ tables


def screen_skill(derived: Path, wanted: list[str], notes: list[str]) -> pd.DataFrame:
    """Per patient, model and horizon, the frozen and the refitted skill of stage 3 (script 40; the refitted arm is
    the model's mechanism of script 41, at the interval the stage ran it with)."""
    days_path = derived / "runs" / "40_stage3_mechanism_runs" / "day_metrics.parquet"
    mech_path = derived / "runs" / "41_stage3_mechanism_analysis" / "mechanisms.yaml"
    if not days_path.exists() or not mech_path.exists():
        notes.append("screening left out: day metrics of script 40 or mechanisms.yaml of script 41 not found")
        return pd.DataFrame()
    mechanisms = (yaml.safe_load(mech_path.read_text()) or {}).get("models", {})
    recommended = {m: (mechanisms.get(m) or {}).get("arm") for m in wanted}  # a twin takes its model's arm
    recommended.update({f"{m}_{TWIN}": arm for m, arm in recommended.items()})
    days = pd.read_parquet(days_path)
    keep = days["model"].eq("persistence") | days["model"].isin([m for m, a in recommended.items() if a])
    return states(pr.patient_skill(days[keep]), recommended, wanted)


def confirmation_skill(derived: Path, results: Path, wanted: list[str],
                       notes: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """The same for every cohort of script 50 (the refitted arm is the recommended arm of its plan)."""
    import json

    days_path = derived / "runs" / "50_confirmation_runs" / "day_metrics.parquet"
    manifest_path = results / "50_confirmation_runs" / "run_manifest.json"
    if not days_path.exists() or not manifest_path.exists():
        notes.append("confirmation left out: day metrics or run manifest of script 50 not found")
        return pd.DataFrame(), []
    settings = json.loads(manifest_path.read_text())["settings"]
    plan = settings["plan"]
    recommended = {m: cf.recommended_arm(plan.get(m) or {}) for m in wanted}
    recommended.update({f"{m}_{TWIN}": cf.recommended_arm(plan.get(f"{m}_{TWIN}") or plan.get(m) or {})
                        for m in wanted})
    days = pd.read_parquet(days_path)
    cohorts = [cf.cohort_key(c["dataset"], c["cohort"]) for c in settings["cohorts"]]
    parts = []
    for cohort in cohorts:
        d = days[(days["cohort"] == cohort) &
                 (days["model"].eq("persistence") | days["model"].isin(list(recommended)))]
        if d["model"].isin(list(recommended)).any():
            parts.append(states(pr.patient_skill(d), recommended, wanted).assign(source=cohort))
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), cohorts


def states(skill: pd.DataFrame, recommended: dict, wanted: list[str]) -> pd.DataFrame:
    """Rows of ``never`` (frozen) and of each model's recommended arm (refitted) of the full models; where a full
    model did not run, its glucose-only twin stands in (``entry``: the full model's name; ``inputs``)."""
    present = set(skill["model"])
    use = {}
    for m in wanted:
        if m in present:
            use[m] = m
        elif f"{m}_{TWIN}" in present:
            use[f"{m}_{TWIN}"] = m
    s = skill[skill["model"].isin(list(use))].copy()
    s["entry"] = s["model"].map(use)
    s["inputs"] = np.where(s["model"].map(is_twin), "glucose only", "all")
    arm = s["model"].map(recommended)
    frozen = s[s["policy"] == "never"].assign(state="frozen")
    refitted = s[arm.notna() & (s["policy"] == arm)].assign(state="refitted")
    return pd.concat([frozen, refitted], ignore_index=True)


def mean_table(skill: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rows = []
    for (source, state, period, model, horizon), g in skill.groupby(
            ["source", "state", "period", "entry", "horizon_min"], sort=False):
        mean, low, high = bootstrap_mean(g["skill"].to_numpy(), replicates, seed)
        rows.append({"source": source, "state": state, "period": period, "model": model, "run": g["model"].iloc[0],
                     "inputs": g["inputs"].iloc[0], "arm": g["policy"].iloc[0],
                     "horizon_min": int(horizon), "patients": int(g["patient_id"].nunique()), "skill": mean,
                     "low": low, "high": high, "rmse": float(g["rmse"].mean())})
    return pd.DataFrame(rows)


def pairwise(skill: pd.DataFrame, dmd: DMDTable, replicates: int, seed: int) -> pd.DataFrame:
    """Every pair of models per source, state and horizon, oriented so that ``model`` has the higher mean skill on
    their common patients (``difference`` >= 0). The DMD is that of the models that ran (a twin's own)."""
    rows = []
    for (source, state, period, horizon), g in skill.groupby(["source", "state", "period", "horizon_min"],
                                                             sort=False):
        run = g.drop_duplicates("entry").set_index("entry")["model"]
        wide = g.pivot_table(index="patient_id", columns="entry", values="skill")
        for a, b in combinations(wide.columns, 2):
            both = wide[[a, b]].dropna()
            if len(both) < 3:
                continue
            if both[a].mean() < both[b].mean():
                a, b = b, a
            d = (both[a] - both[b]).to_numpy()
            mean, low, high = bootstrap_mean(d, replicates, seed)
            floor = dmd.pair(period, int(horizon), run[a], run[b]) if len(dmd) else np.nan
            verdict = "tie" if np.isfinite(floor) and abs(mean) < floor else \
                "better" if low > 0 else "worse" if high < 0 else "uncertain"
            rows.append({"source": source, "state": state, "period": period, "horizon_min": int(horizon),
                         "model": a, "against": b, "patients": int(len(d)), "difference": mean, "low": low,
                         "high": high, "patients_better": int((d > 0).sum()), "dmd": floor, "verdict": verdict})
    return pd.DataFrame(rows)


def leader_table(skill: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Per source, state and horizon: the model with the highest mean on the patients every model has, and the
    models whose comparison with it is a tie or uncertain."""
    rows = []
    for (source, state, horizon), g in skill.groupby(["source", "state", "horizon_min"], sort=False):
        wide = g.pivot_table(index="patient_id", columns="entry", values="skill").dropna()
        if wide.shape[1] < 2:
            continue
        if len(wide) < 3:
            continue
        order = wide.mean().sort_values(ascending=False)
        best = order.index[0]
        p = pairs[(pairs["source"] == source) & (pairs["state"] == state) & (pairs["horizon_min"] == horizon)]
        with_best = p[(p["model"] == best) | (p["against"] == best)]
        level = [m for m in order.index[1:]
                 if not with_best[((with_best["model"] == m) | (with_best["against"] == m)) &
                                  (with_best["verdict"] == "better")].shape[0]]
        inputs = "glucose only" if (g["inputs"] == "glucose only").any() else "all"
        rows.append({"source": source, "state": state, "horizon_min": int(horizon), "inputs": inputs,
                     "patients": int(len(wide)),
                     "best": best, "best_skill": float(order.iloc[0]), "not_worse": " ".join(level),
                     "ranking": " > ".join(order.index)})
    return pd.DataFrame(rows)


# ============================================================================ figure


def plot_reference(pairs: pd.DataFrame, reference: str, models: list[str], sources: list[str]):
    """Rows: frozen, refitted; columns: horizons. Every other model minus the reference model (points, 95 %), one
    marker per source."""
    horizons = sorted(pairs["horizon_min"].unique())
    others = [m for m in models if m != reference]
    fig, axes = model_axes(len(STATES) * len(horizons), 3.6, 0.45 * len(others) + 1.2, ncols=len(horizons))
    for r, state in enumerate(STATES):
        for c, horizon in enumerate(horizons):
            ax = axes[r * len(horizons) + c]
            p = pairs[(pairs["state"] == state) & (pairs["horizon_min"] == horizon)]
            for k, source in enumerate(sources):
                s = p[p["source"] == source]
                for i, m in enumerate(others):
                    row = s[((s["model"] == m) & (s["against"] == reference)) |
                            ((s["model"] == reference) & (s["against"] == m))]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    sign = 1 if row["model"] == m else -1
                    mid, lo, hi = sign * row["difference"], sign * row["low"], sign * row["high"]
                    lo, hi = min(lo, hi), max(lo, hi)
                    y = i + (k - (len(sources) - 1) / 2) * 0.12
                    ax.errorbar(100 * mid, y, xerr=[[100 * (mid - lo)], [100 * (hi - mid)]], fmt=MARKERS[k % 10],
                                color=color(m), ms=4, lw=0.9, mfc=color(m) if k == 0 else "white",
                                label=source if (r, c, i) == (0, 0, 0) else None)
            ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
            ax.set_yticks(range(len(others)), [label(m) for m in others] if c == 0 else [""] * len(others))
            ax.invert_yaxis()
            ax.set_xlabel(f"skill minus {label(reference)} (points)")
            ax.set_title(f"{state}, {horizon} min", loc="left", fontsize=9)
    handles = [plt_handle(MARKERS[k % 10], k == 0, s) for k, s in enumerate(sources)]
    axes[0].legend(handles=handles, fontsize=6, loc="lower right")
    fig.suptitle(f"Every model against {label(reference)}, frozen and refitted as recommended", fontweight="bold",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig


def plt_handle(marker: str, filled: bool, name: str):
    from matplotlib.lines import Line2D

    return Line2D([], [], marker=marker, ls="none", color=TEXT_SECONDARY, mfc=TEXT_SECONDARY if filled else "white",
                  ms=4, label=name)


# ============================================================================ summary


def write_summary(means, pairs, leaders, sources, shown, models, reference, notes, release, resources) -> str:
    lines = ["# 74 - Which model family forecasts best, frozen and refitted", "", f"Code release {release}.",
             f"Machine: {resources.describe()}",
             f"Models {models}; sources {sources}. Frozen = never; refitted = the model's mechanism at its optimal "
             "interval (ARX: at every sample). Paired per patient; tie below the DMD of the pair (D75). D103.", ""]
    lines += ["## Warnings", ""] + ([f"- {n}" for n in notes] or ["- none"]) + [""]
    if not leaders.empty:
        t = leaders.copy()
        t["best"] = t["best"].map(label)
        t["not_worse"] = t["not_worse"].map(lambda s: ", ".join(label(m) for m in s.split()) if s else "")
        t["best_skill"] = (100 * t["best_skill"]).round(2)
        lines += ["## Best model and those that do not lose to it (mean skill on the common patients, %)", "",
                  markdown_table(t[["source", "inputs", "state", "horizon_min", "patients", "best", "best_skill",
                                    "not_worse"]]), ""]
    if not pairs.empty and reference in models:
        p = pairs[(pairs["model"] == reference) | (pairs["against"] == reference)].copy()
        p["other"] = np.where(p["model"] == reference, p["against"], p["model"])
        sign = np.where(p["model"] == reference, -1, 1)
        p["cell"] = [f"{100 * s * d:+.2f} ({100 * min(s * lo, s * hi):+.2f} to {100 * max(s * lo, s * hi):+.2f}; "
                     f"{flip(v) if s < 0 else v})" for s, d, lo, hi, v in
                     zip(sign, p["difference"], p["low"], p["high"], p["verdict"])]
        for source in shown:
            q = p[p["source"] == source]
            if q.empty:
                continue
            table = q.pivot_table(index=["state", "other"], columns="horizon_min", values="cell", aggfunc="first")
            table.columns = [f"{h} min" for h in table.columns]
            table = table.reset_index()
            table["other"] = table["other"].map(label)
            lines += [f"## {source}: every model minus {label(reference)} (skill points, 95 %; verdict of the model)",
                      "", markdown_table(table), ""]
        lines += ["Every pair, every cohort: `pairwise.csv`.", ""]
    if not means.empty:
        m = means[means["source"].isin(shown)].copy()
        m["cell"] = [f"{100 * s:.2f}" for s in m["skill"]]
        table = m.pivot_table(index=["source", "state", "model"], columns="horizon_min", values="cell",
                              aggfunc="first")
        table.columns = [f"{h} min (%)" for h in table.columns]
        lines += ["## Mean skill", "", markdown_table(table.reset_index()), ""]
    lines += ["## Checks to look at", "",
              "- `family_comparison.png`: whether the order of the families holds from the screening to the cohorts.",
              ""]
    return "\n".join(lines)


def flip(verdict: str) -> str:
    """The verdict seen from the other model of the pair."""
    return {"better": "worse", "worse": "better"}.get(verdict, verdict)


if __name__ == "__main__":
    main()
