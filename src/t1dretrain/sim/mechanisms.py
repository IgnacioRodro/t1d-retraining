"""Stage 3 for one patient, anchor and model (script 40; HANDOFF 7.5, D86): how a model takes in new data.

Every arm refits at the anchor and every ``interval`` days (the model's
optimal interval of script 31, D84) on the growing window (stage 2), and
differs only in how a refit uses what came before. An arm can name its own
interval with the suffix ``_every<days>`` (``restart_every3``, ``warm_every1``;
the confirmation, script 50); without it, the job's ``interval`` applies.

- ``never``: the fit at the anchor only (the population model fine-tuned on
  the 14 days before it);
- ``restart``: every refit starts again from the population model (D65; the
  growing window of stage 2);
- ``restart_all`` / ``restart_head`` (networks): the same, fine-tuning every
  layer / only the output layers; ``restart_other``: the other choice than the
  one of script 11 (``finetune.train`` of ``choices.yaml``);
- ``warm``: every refit continues the previous personal model on the growing
  window (LightGBM: new trees; networks: more training): a warm start;
- ``leaves_d<decay>`` (LightGBM): the trees of the previous personal model are
  kept and their leaf values re-estimated on the growing window, new value =
  decay x old + (1 - decay) x refitted (D88: 0.9, 0.5 and 0, a full re-estimate);
- ``scratch`` (LightGBM): every refit is a personal model from scratch on the
  growing window (no population model);
- ``online_l<λ>`` (ARX): the update at every sample (D82) with forgetting
  factor λ (effective memory about 1 / (1 - λ) samples of 5 min);
- ``incremental_w<days>`` and ``incremental_w<days>_t<alpha>``: the previous
  personal model continued on the last ``days`` days only (a rolling window,
  not the growing one), as the incrementally retrained stacked LSTM of Shen and
  Kleinberg (2025; script 75, D103). With ``_t<alpha>`` (their "parameter
  transfer") a refit is skipped, and the model in use kept, when a two-sample
  Welch t-test does not tell the glucose of those days from that of the
  ``days`` before them (p > alpha).

``never``, ``restart``, ``warm``, ``leaves`` and ``incremental`` share the fit at the anchor. A
refit that fails keeps the model in use, as in stage 2 (at the anchor, the fit
of never); if the fit at the anchor fails, the patient and anchor are left out
for every arm of the model, so that all arms are scored on the same patients
and origins. The cost of an arm counts every fit it uses (``compute_seconds``:
networks without the wait for a GPU turn); ``weighting`` names the arm of each
fit (``anchor`` for the shared one), so that two arms' fits are never taken for
the same fit.
"""

from __future__ import annotations

import copy
import time
import traceback

import numpy as np
import pandas as pd

from .policies import JobData, PolicyResult, run_online
from .stage0 import is_learned, make_model, split_name, split_variant

ANCHOR_ARMS = ("never", "restart", "warm", "leaves")  # kinds of arm that start from the fit at the anchor
CONTINUED = ("warm", "leaves", "incremental")  # kinds whose refits change a copy of the model in use
FIT_ARMS = ("restart", "restart_all", "restart_head", "restart_other", "scratch")  # every refit is a new model
CACHE_VERSION = 1  # bump when the simulation changes, so that script 40 recomputes its cached results


def split_interval(arm: str) -> tuple[str, int | None]:
    """('restart', 3) for 'restart_every3'; ('restart', None) for 'restart' (the job's interval applies)."""
    head, sep, tail = arm.rpartition("_every")
    if sep and head and tail.isdigit() and int(tail) > 0:
        return head, int(tail)
    return arm, None


def parse_arm(arm: str) -> tuple[str, object]:
    """The kind of an arm and its value: ('online', 0.995) for 'online_l0.995', ('leaves', 0.5) for 'leaves_d0.5',
    ('incremental', (7, 0.05)) for 'incremental_w7_t0.05' and ('incremental', (7, None)) for 'incremental_w7',
    ('warm', None) for 'warm' and for 'warm_every3'. ``never`` and the online update have no refit interval."""
    arm, interval = split_interval(arm)
    if interval is not None and (arm == "never" or arm.startswith("online_l")):
        raise ValueError(f"arm {arm!r} has no refit interval")
    if arm.startswith("incremental_w"):
        days, _, test = arm[len("incremental_w"):].partition("_t")
        try:
            window, alpha = int(days), (float(test) if test else None)
        except ValueError:
            raise ValueError(f"arm {arm!r}: incremental_w<days> or incremental_w<days>_t<alpha>") from None
        if window < 1 or (alpha is not None and not 0 < alpha < 1):
            raise ValueError(f"arm {arm!r}: the window must be at least a day and alpha in (0, 1)")
        return "incremental", (window, alpha)
    for prefix, kind in (("online_l", "online"), ("leaves_d", "leaves")):
        if arm.startswith(prefix):
            value = float(arm[len(prefix):])
            if not (0 < value <= 1 if kind == "online" else 0 <= value <= 1):
                raise ValueError(f"arm {arm!r}: " + ("the forgetting factor must be in (0, 1]" if kind == "online"
                                                     else "the decay must be in [0, 1]"))
            return kind, value
    if arm in ANCHOR_ARMS + FIT_ARMS and arm != "leaves":  # a leaf refit always names its decay
        return arm, None
    plain = [a for a in ANCHOR_ARMS + FIT_ARMS if a != "leaves"]
    raise ValueError(f"unknown arm {arm!r}: one of {plain}, online_l<forgetting factor>, leaves_d<decay> or "
                     "incremental_w<days>[_t<alpha>]")


def arm_model(name: str, arm: str, settings: dict) -> tuple[str, dict]:
    """The model name and settings of an arm's fits: the population model with the other fine-tuning (networks),
    the personal model from scratch (LightGBM), or the model itself."""
    family = split_name(name)[0]
    kind, value = parse_arm(arm)
    if kind in ("restart_all", "restart_head", "restart_other"):
        chosen = dict((settings.get("finetune") or {}).get(family) or {})
        train = kind.split("_")[1] if kind != "restart_other" else ("head" if chosen.get("train") == "all" else "all")
        finetune = {**settings.get("finetune", {}), family: {**chosen, "train": train}}
        return name, {**settings, "finetune": finetune}
    if kind == "scratch":
        base, variant = split_variant(name)
        return family + (f"_{variant}" if variant else ""), settings
    if kind == "online":
        base = split_variant(name)[0]
        return name, {**settings, base: {**(settings.get(base) or {}), "forgetting": value}}
    return name, settings


def glucose_changed(series, day: pd.Timestamp, days: int, alpha: float) -> tuple[bool, float]:
    """Parameter transfer of the incremental arm: the glucose of the ``days`` days before ``day`` against that of
    the ``days`` before them (two-sample Welch t-test on the grid values). A refit only when they differ (p <= alpha);
    too few values to test means a refit."""
    from scipy.stats import ttest_ind

    span = pd.Timedelta(days=int(days))
    recent = series.between(day - span, day)[0]
    before = series.between(day - 2 * span, day - span)[0]
    a, b = recent[np.isfinite(recent)], before[np.isfinite(before)]
    if len(a) < 3 or len(b) < 3:
        return True, float("nan")
    p = float(ttest_ind(a, b, equal_var=False).pvalue)
    return (not np.isfinite(p)) or p <= alpha, p


def fit_days_of(anchor: pd.Timestamp, test_days: int, interval: int) -> list[pd.Timestamp]:
    return [anchor + pd.Timedelta(days=k) for k in range(0, int(test_days), max(1, int(interval)))]


def log_row(data: JobData, arm: str, seed: int, fit_day: pd.Timestamp, train, log, rows: np.ndarray,
            started: float) -> dict:
    details = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in log.details.items()
               if k not in ("curve", "compute_seconds")}
    seconds = round(time.time() - started, 3)
    return data.row(arm, seed, fit_day=fit_day.date(), train_start=data.base_start.date(), weighting=arm,
                    train_days=int(len(np.unique(train.samples.day))), train_samples=int(train.samples.n),
                    fit_samples=log.n_samples, fit_seconds=seconds,
                    compute_seconds=float(log.details.get("compute_seconds", seconds)), test_origins=int(len(rows)),
                    **details)


def evaluate_mechanisms(job: dict) -> PolicyResult:
    """Every arm of one model for one patient and anchor (see the module docstring)."""
    started = time.time()
    patient_id, period, anchor_day, name = job["patient_id"], job["period"], int(job["anchor_day"]), job["model"]
    try:
        try:
            data = JobData(job)
        except LookupError as error:
            return PolicyResult(patient_id, period, anchor_day, name, pd.DataFrame(), None, error=str(error))
        s, test, anchor, end = data.settings, data.test, data.anchor, data.end
        device = job.get("device", "cpu")
        seed = int(s["seeds"][0]) if is_learned(name) else 0
        info, refits = [], {}
        if job.get("with_reference", True):
            data.record_reference()
        arms = list(job["arms"])
        kinds = {arm: parse_arm(arm)[0] for arm in arms}
        values = {arm: parse_arm(arm)[1] for arm in arms}

        # the fit at the anchor (shared by never, restart, warm and leaves): without it the model is left out
        begin = time.time()
        model = make_model(name, seed, s, device)
        train = data.train_data(data.base_start, anchor)
        try:
            log = model.timed_fit(train)
            pred = model.predict(test)
            anchor_fit = (model, pred, log_row(data, "anchor", seed, anchor, train, log, np.arange(test.n), begin))
        except Exception as error:  # noqa: BLE001
            message = f"{type(error).__name__}: {str(error).strip()}"[:300]
            for arm in arms:
                info.append(data.row(arm, seed, fit_day=anchor.date(), train_start=data.base_start.date(),
                                     weighting="anchor", test_origins=0, error=message,
                                     excluded="the fit at the anchor failed"))
                refits[(arm, seed)] = []
            arms = []

        for arm in arms:
            kind = kinds[arm]
            if kind == "online":
                own_name, own = arm_model(name, arm, s)
                info.append({**run_online(data, arm, seed, make_model(own_name, seed, own, "cpu")), "weighting": arm})
                refits[(arm, seed)] = [anchor]
                continue
            used_days, current = [], None
            interval = split_interval(arm)[1] or job.get("interval")
            if kind != "never" and not interval:
                raise ValueError(f"arm {arm!r} needs a refit interval (the job's or its own _every<days>)")
            days = [anchor] if kind == "never" else fit_days_of(anchor, data.test_days, int(interval))
            for i, day in enumerate(days):
                stop = days[i + 1] if i + 1 < len(days) else end
                rows = np.flatnonzero((data.test_day >= day) & (data.test_day < stop))
                if i == 0 and (kind in ANCHOR_ARMS or kind == "incremental"):  # the shared fit at the anchor
                    model, pred_all, row = anchor_fit
                    current = copy.deepcopy(model) if kind in CONTINUED else model
                    info.append({**row, "policy": arm, "test_origins": int(len(rows))})
                    used_days.append(day)
                    if len(rows):
                        data.record(arm, seed, day, rows, pred_all[rows])
                    continue
                begin = time.time()
                if kind == "incremental":  # the last days only; with a test, skipped when the glucose did not change
                    window, alpha = values[arm]
                    train = data.train_data(max(data.base_start, day - pd.Timedelta(days=window)), day)
                    changed, p_value = (True, float("nan")) if alpha is None else \
                        glucose_changed(data.series, day, window, alpha)
                    if not changed:
                        info.append(data.row(arm, seed, fit_day=day.date(), train_start=data.base_start.date(),
                                             weighting=arm, test_origins=int(len(rows)), skipped=True,
                                             p_value=p_value, compute_seconds=0.0,
                                             served_by=str(used_days[-1].date()) if used_days else "anchor"))
                        if len(rows) and current is not None:
                            data.record(arm, seed, used_days[-1], rows, current.predict(test.subset(rows)))
                        continue
                else:
                    train, p_value = data.train_data(data.base_start, day), None
                try:
                    if kind in FIT_ARMS:
                        own_name, own = arm_model(name, arm, s)
                        model = make_model(own_name, seed, own, device)
                        log = model.timed_fit(train)
                    elif kind in ("warm", "incremental"):
                        model = current
                        log = model.continue_fit(train)
                    else:  # leaves
                        model = current
                        log = model.refit_leaves(train, values[arm])
                    info.append({**log_row(data, arm, seed, day, train, log, rows, begin),
                                 **({"p_value": p_value, "skipped": False} if kind == "incremental" else {})})
                    current, used = model, day
                    used_days.append(day)
                except Exception as error:  # noqa: BLE001  (keeps the model in use, as in stage 2)
                    info.append(data.row(arm, seed, fit_day=day.date(), train_start=data.base_start.date(),
                                         weighting=arm, test_origins=int(len(rows)),
                                         served_by=str(used_days[-1].date()) if used_days else "anchor",
                                         error=f"{type(error).__name__}: {str(error).strip()}"[:300]))
                    if current is None:  # an arm's own first fit failed: the fit of never serves
                        current, used = anchor_fit[0], anchor
                    else:
                        used = used_days[-1] if used_days else day
                if len(rows) and current is not None:
                    data.record(arm, seed, used, rows, current.predict(test.subset(rows)))
            refits[(arm, seed)] = used_days
        refits = {f"{p}|{seed}": [d.date().isoformat() for d in v] for (p, seed), v in refits.items()}
        return PolicyResult(patient_id, period, anchor_day, name, data.days(), data.clarke, info, refits,
                            round(time.time() - started, 2))
    except Exception:  # noqa: BLE001
        return PolicyResult(patient_id, period, anchor_day, name, pd.DataFrame(), None,
                            seconds=round(time.time() - started, 2), error=traceback.format_exc(limit=6))
