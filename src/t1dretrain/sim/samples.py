"""Forecasting samples from a patient's feature grid (script 05 output).

A sample is one forecast origin: the inputs of the ``lookback`` hours that end
at the origin slot and the readings ``horizons`` minutes later. The rules
(D58) are the same for every model, so all models are scored on the same
origins:

- the origin is a slot with a real reading that is a valid target
  (``target_ok``), on an allowed day, on the stride (absolute slot index
  multiple of stride / step, so the origins do not depend on the model);
- the whole input window lies in the origin's segment (no clock break);
- the last ``recent_complete_min`` minutes of inputs are complete and at
  least ``min_window_share`` of the window has an input;
- optional slot conditions (e.g. ``insulin_known``) hold at the origin;
- a target is the reading at origin + horizon if it is a valid target in the
  same segment, else NaN (that horizon is skipped for that origin).

Inputs are oldest first; the last column is the origin. Interpolated inputs
only fill gaps closed before the origin, because origins are real readings
(D55).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Samples:
    patient_id: str
    step_min: float
    horizons_min: list[int]
    origin_slot: np.ndarray
    origin_ts: np.ndarray
    day: np.ndarray
    inputs: np.ndarray
    targets: np.ndarray
    exog: dict[str, np.ndarray] = field(default_factory=dict)
    trajectory: np.ndarray | None = None  # every slot up to the longest horizon (for models trained on the path)
    patient: np.ndarray | None = None     # patient of every sample (samples of several patients, see concat_samples)
    context: np.ndarray | None = None     # longer glucose history for context-only models (NaN outside the segment)
    context_exog: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.origin_slot)

    @property
    def horizon_steps(self) -> list[int]:
        return [int(round(h / self.step_min)) for h in self.horizons_min]

    def subset(self, mask: np.ndarray) -> "Samples":
        return Samples(self.patient_id, self.step_min, self.horizons_min, self.origin_slot[mask], self.origin_ts[mask],
                       self.day[mask], self.inputs[mask], self.targets[mask], {k: v[mask] for k, v in self.exog.items()},
                       None if self.trajectory is None else self.trajectory[mask],
                       None if self.patient is None else self.patient[mask],
                       None if self.context is None else self.context[mask],
                       {k: v[mask] for k, v in self.context_exog.items()})


def known_before(samples: Samples, end) -> np.ndarray:
    """Samples whose every target (up to the longest horizon) is earlier than ``end`` (HANDOFF 7.1: a model fitted
    at the start of day d learns only from targets before d, so no glucose value of a test day is a training
    target)."""
    last = np.asarray(samples.origin_ts).astype("datetime64[ns]") + np.timedelta64(int(max(samples.horizons_min)), "m")
    return last < np.datetime64(pd.Timestamp(end).to_datetime64(), "ns")


ARRAYS = ("origin_slot", "origin_ts", "day", "inputs", "targets", "trajectory", "patient")


def concat_samples(parts: list[Samples], name: str = "pool") -> Samples:
    """Samples of several patients in one set (population models). All parts must share step, horizons and window."""
    parts = [p for p in parts if p.n]
    if not parts:
        raise ValueError("no samples to concatenate")
    first = parts[0]
    for p in parts[1:]:
        if p.step_min != first.step_min or p.horizons_min != first.horizons_min or \
                p.inputs.shape[1] != first.inputs.shape[1]:
            raise ValueError(f"samples of {p.patient_id} differ in step, horizons or window from {first.patient_id}")
    patient = np.concatenate([p.patient if p.patient is not None else np.full(p.n, p.patient_id, dtype=object)
                              for p in parts])
    trajectory = None if any(p.trajectory is None for p in parts) else np.concatenate([p.trajectory for p in parts])
    return Samples(name, first.step_min, list(first.horizons_min), np.concatenate([p.origin_slot for p in parts]),
                   np.concatenate([p.origin_ts for p in parts]), np.concatenate([p.day for p in parts]),
                   np.concatenate([p.inputs for p in parts]), np.concatenate([p.targets for p in parts]),
                   {k: np.concatenate([p.exog[k] for p in parts]) for k in first.exog}, trajectory, patient)


def save_samples(samples: Samples, folder) -> None:
    """One .npy file per array (read back with memory mapping by ``load_samples``)."""
    import json
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ARRAYS:
        value = getattr(samples, name)
        if value is not None:
            np.save(folder / f"{name}.npy", value.astype(str) if name == "patient" else value, allow_pickle=False)
    for name, value in samples.exog.items():
        np.save(folder / f"exog_{name}.npy", value, allow_pickle=False)
    (folder / "meta.json").write_text(json.dumps({"patient_id": samples.patient_id, "step_min": samples.step_min,
                                                  "horizons_min": list(samples.horizons_min),
                                                  "exog": list(samples.exog)}))


def load_samples(folder, mmap: bool = True) -> Samples:
    import json
    from pathlib import Path

    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    mode = "r" if mmap else None
    arrays = {name: np.load(folder / f"{name}.npy", mmap_mode=mode) if (folder / f"{name}.npy").exists() else None
              for name in ARRAYS}
    exog = {name: np.load(folder / f"exog_{name}.npy", mmap_mode=mode) for name in meta["exog"]}
    return Samples(meta["patient_id"], float(meta["step_min"]), list(meta["horizons_min"]), arrays["origin_slot"],
                   arrays["origin_ts"], arrays["day"], arrays["inputs"], arrays["targets"], exog, arrays["trajectory"],
                   arrays["patient"])


class DenseGrid:
    """A patient's grid on contiguous slots (slots of days without readings filled with NaN / False)."""

    def __init__(self, grid: pd.DataFrame) -> None:
        self.patient_id = str(grid["patient_id"].iloc[0])
        self.slot0 = int(grid["slot"].min())
        self.n = int(grid["slot"].max()) - self.slot0 + 1
        self.position = grid["slot"].to_numpy() - self.slot0
        steps = np.diff(grid["ts"].to_numpy("datetime64[ns]").astype(np.int64)) / np.maximum(np.diff(grid["slot"]), 1)
        self.step_min = float(np.median(steps) / 60e9) if len(steps) else 5.0
        self.ts0 = grid["ts"].iloc[0] - pd.Timedelta(minutes=self.step_min * (int(grid["slot"].iloc[0]) - self.slot0))
        self._grid = grid
        segment = np.full(self.n, np.nan)
        segment[self.position] = grid["segment"].to_numpy()
        self.segment = pd.Series(segment).ffill().bfill().to_numpy()

    def column(self, name: str, fill=np.nan) -> np.ndarray:
        values = self._grid[name].to_numpy()
        dtype = values.dtype if values.dtype != bool else bool
        out = np.full(self.n, fill, dtype=float if dtype != bool else bool)
        out[self.position] = values
        return out

    def slot_times(self) -> np.ndarray:
        return (self.ts0 + pd.to_timedelta(np.arange(self.n) * self.step_min, unit="min")).to_numpy()


def make_samples(grid: pd.DataFrame, horizons_min: list[int], lookback_h: float, stride_min: float,
                 allowed_days: set | None = None, min_window_share: float = 0.8, recent_complete_min: float = 30,
                 origin_conditions: tuple[str, ...] = (), exog_columns: tuple[str, ...] = (),
                 trajectory: bool = False, context_h: float | None = None,
                 context_exog: tuple[str, ...] = ()) -> Samples:
    """Samples of one patient (see the module docstring). ``context_h`` adds a longer history for context-only
    models (``Samples.context``: the glucose input of the ``context_h`` hours up to the origin, NaN before the
    origin's segment; it does not change which origins are kept)."""
    dense = DenseGrid(grid)
    step = dense.step_min
    lookback = int(round(lookback_h * 60 / step))
    recent = int(round(recent_complete_min / step)) + 1
    stride = max(1, int(round(stride_min / step)))
    shifts = [int(round(h / step)) for h in horizons_min]
    if any(abs(s * step - h) > 1e-6 for s, h in zip(shifts, horizons_min)):
        raise ValueError(f"horizons {horizons_min} are not multiples of the {step}-min step")

    inputs = dense.column("glucose_input")
    target_ok = dense.column("target_ok", fill=False)
    glucose = dense.column("glucose")
    times = dense.slot_times()
    days = times.astype("datetime64[D]")
    ok = target_ok.copy()
    ok &= ((np.arange(dense.n) + dense.slot0) % stride) == 0
    ok[: lookback - 1] = False
    for name in origin_conditions:
        ok &= dense.column(name, fill=False).astype(bool)
    if allowed_days is not None:
        allowed = np.array(sorted(pd.DatetimeIndex(list(allowed_days)).values.astype("datetime64[D]")))
        ok &= np.isin(days, allowed)
    candidates = np.flatnonzero(ok)
    if candidates.size:
        start = candidates - lookback + 1
        same_segment = dense.segment[start] == dense.segment[candidates]
        finite = np.isfinite(inputs)
        csum = np.concatenate([[0], np.cumsum(finite)])
        window_share = (csum[candidates + 1] - csum[start]) / lookback
        recent_ok = (csum[candidates + 1] - csum[candidates + 1 - recent]) == recent
        candidates = candidates[same_segment & (window_share >= min_window_share) & recent_ok]
    index = candidates[:, None] + np.arange(-lookback + 1, 1)[None, :]

    def ahead(shift: int) -> np.ndarray:
        t = candidates + shift
        inside = t < dense.n
        valid = np.zeros(len(candidates), dtype=bool)
        valid[inside] = target_ok[t[inside]] & (dense.segment[t[inside]] == dense.segment[candidates[inside]])
        out = np.full(len(candidates), np.nan, dtype=np.float32)
        out[valid] = glucose[t[valid]]
        return out

    targets = np.stack([ahead(shift) for shift in shifts], axis=1) if len(candidates) else \
        np.zeros((0, len(shifts)), dtype=np.float32)
    path = None
    if trajectory:
        path = np.stack([ahead(k) for k in range(1, max(shifts) + 1)], axis=1) if len(candidates) else \
            np.zeros((0, max(shifts)), dtype=np.float32)
    exog = {name: dense.column(name)[index].astype(np.float32) for name in exog_columns}
    context, context_extra = None, {}
    if context_h:
        length = int(round(context_h * 60 / step))
        far = candidates[:, None] + np.arange(-length + 1, 1)[None, :]
        inside = far >= 0
        far_safe = np.where(inside, far, 0)
        inside &= dense.segment[far_safe] == dense.segment[candidates][:, None]
        context = np.where(inside, inputs[far_safe], np.nan).astype(np.float32)
        context_extra = {name: np.where(inside, dense.column(name)[far_safe], np.nan).astype(np.float32)
                         for name in context_exog}
    return Samples(dense.patient_id, step, list(horizons_min), candidates + dense.slot0, times[candidates],
                   days[candidates], inputs[index].astype(np.float32), targets, exog, path, None, context, context_extra)


def training_series(grid: pd.DataFrame, allowed_days: set) -> tuple[np.ndarray, np.ndarray, float]:
    """The patient's input glucose on contiguous slots, NaN outside ``allowed_days``, with the segment of
    every slot (AR-type models must not pair readings across a clock break)."""
    dense = DenseGrid(grid)
    values = dense.column("glucose_input")
    days = dense.slot_times().astype("datetime64[D]")
    allowed = np.array(sorted(pd.DatetimeIndex(list(allowed_days)).values.astype("datetime64[D]")))
    values[~np.isin(days, allowed)] = np.nan
    return values, dense.segment, dense.step_min
