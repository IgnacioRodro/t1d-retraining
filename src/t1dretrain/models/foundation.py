"""Foundation models used zero-shot: the "context only" mechanism (HANDOFF 6.1, M17-M19; D70).

They are not trained on the patient: each forecast reads a long glucose
history ending at the origin (``context_hours``, from ``Samples.context``)
and the model's pretrained weights do the rest. Refitting does not apply;
what can be changed is how much history they read (stage 3, Q3).

- ``chronos2``: Chronos-2 (amazon/chronos-2, chronos-forecasting 2.3.2) with
  IOB and COB as past covariates and the time of day as a known future
  covariate; median of the predicted quantiles.
- ``chronos2_univariate``: the same model reading glucose only (what the
  covariates add).
- ``timesfm``: TimesFM 2.5 (google/timesfm-2.5-200m-pytorch, Apache-2.0;
  timesfm 3.0.2), glucose only; point forecast. Missing values are
  interpolated by the library.
- ``ttm``: Tiny Time Mixers r2 (ibm-granite/granite-timeseries-ttm-r2,
  granite-tsfm 0.3.11), glucose only, with the context length of the
  checkpoint (512 slots of 5 min); missing values carried forward. The
  checkpoint is named in the settings (``revision``, D77): ``512-96-r2``, the
  r2 model for minutely to hourly data. Without it the library's own choice
  for a 72-h context and a 2-h horizon is ``512-48-ft-r2.1``, a model of the
  r2.1 release (made for daily and weekly data, with a frequency token), which
  did worse than persistence at 30 min (it. 20b).

Weights are downloaded from the Hugging Face Hub the first time (cached in the
usual Hugging Face folder) and loaded once per process and device.
"""

from __future__ import annotations

import numpy as np

from ..sim.samples import Samples
from .base import FitLog, ForecastModel, TrainData, clip

FOUNDATION_MODELS = {"chronos2": "amazon/chronos-2", "chronos2_univariate": "amazon/chronos-2",
                     "timesfm": "google/timesfm-2.5-200m-pytorch", "ttm": "ibm-granite/granite-timeseries-ttm-r2"}
COVARIATES = ("iob", "cob")  # past covariates of Chronos-2 (also in Samples.context_exog)
DEFAULT_CONTEXT_HOURS = 72.0
TTM_REVISIONS = {"512-96-r2": "main"}  # TTM checkpoints whose branch on the Hub is not named after them
_LOADED: dict = {}  # (kind, model id, revision, device, context slots, path steps) -> backend, once per process
_FAILED: dict = {}  # same key -> error message: a model that failed to load is not retried in this process


def _fill(x: np.ndarray) -> np.ndarray:
    """Carry values forward along each row; leading gaps take the first value."""
    out = x.astype(np.float32).copy()
    for j in range(1, out.shape[1]):
        gap = ~np.isfinite(out[:, j])
        out[gap, j] = out[gap, j - 1]
    for j in range(out.shape[1] - 2, -1, -1):
        gap = ~np.isfinite(out[:, j])
        out[gap, j] = out[gap, j + 1]
    return out


def _hours(samples: Samples, slots: np.ndarray) -> np.ndarray:
    """Hour of the day of slots relative to the origin (``slots`` = 0 at the origin, negative before)."""
    origin = np.asarray(samples.origin_ts).astype("datetime64[ns]").astype(np.int64)
    t = origin[:, None] + slots[None, :] * samples.step_min * 60e9
    return (t % 86_400e9) / 3_600e9


class Chronos2Backend:
    def __init__(self, pipeline, covariates: bool, batch_size: int) -> None:
        self.pipeline, self.covariates, self.batch_size = pipeline, covariates, batch_size

    def forecast(self, samples: Samples, context: np.ndarray, steps: int) -> np.ndarray:
        n, length = context.shape
        inputs = []
        if self.covariates:
            past_hours = _hours(samples, np.arange(-length + 1, 1))
            future_hours = _hours(samples, np.arange(1, steps + 1))
            exog = {name: np.nan_to_num(samples.context_exog[name][:, -length:], nan=0.0) for name in COVARIATES}
        for i in range(n):
            item = {"target": context[i]}
            if self.covariates:
                item["past_covariates"] = {**{name: exog[name][i] for name in COVARIATES},
                                           "hour_sin": np.sin(2 * np.pi * past_hours[i] / 24),
                                           "hour_cos": np.cos(2 * np.pi * past_hours[i] / 24)}
                item["future_covariates"] = {"hour_sin": np.sin(2 * np.pi * future_hours[i] / 24),
                                             "hour_cos": np.cos(2 * np.pi * future_hours[i] / 24)}
            inputs.append(item)
        median = self.pipeline.quantiles.index(0.5)
        out = self.pipeline.predict(inputs, prediction_length=steps, batch_size=self.batch_size,
                                    context_length=length)
        return np.stack([np.asarray(p[0, median, :steps].detach().cpu().float().numpy()) for p in out])


class TimesFMBackend:
    def __init__(self, model) -> None:
        self.model = model

    def forecast(self, samples: Samples, context: np.ndarray, steps: int) -> np.ndarray:
        inputs = [row[np.argmax(np.isfinite(row)):].astype(np.float64) for row in context]  # library interpolates
        point, _ = self.model.forecast(horizon=steps, inputs=list(inputs))
        return np.asarray(point)[:, :steps]


class TTMBackend:
    def __init__(self, model, device: str, batch_size: int) -> None:
        self.model, self.device, self.batch_size = model, device, batch_size
        self.context_slots = int(model.config.context_length)

    def forecast(self, samples: Samples, context: np.ndarray, steps: int) -> np.ndarray:
        import torch

        x = _fill(context[:, -self.context_slots:])
        if x.shape[1] < self.context_slots:  # shorter history than the checkpoint: repeat the first value
            x = np.concatenate([np.repeat(x[:, :1], self.context_slots - x.shape[1], axis=1), x], axis=1)
        token = None
        if getattr(self.model.config, "resolution_prefix_tuning", False):  # checkpoints trained with a frequency token
            from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING

            token = DEFAULT_FREQUENCY_MAPPING.get(f"{int(round(samples.step_min))}min", 0)
        out = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                batch = torch.from_numpy(x[start:start + self.batch_size, :, None]).to(self.device)
                extra = {} if token is None else {"freq_token": torch.full((len(batch),), token, dtype=torch.long,
                                                                           device=self.device)}
                out.append(self.model(past_values=batch, **extra).prediction_outputs[:, :steps, 0].float().cpu()
                           .numpy())
        return np.concatenate(out) if out else np.zeros((0, steps), dtype=np.float32)


def scaled_batch(batch_size: int, context_hours: float) -> int:
    """Forecasts per batch. ``batch_size`` is set for the default context (72 h); a longer context takes
    proportionally fewer, so that a batch needs about the same GPU memory (script 43 reads up to 14 days)."""
    return max(1, min(batch_size, int(batch_size * DEFAULT_CONTEXT_HOURS / max(context_hours, 1.0))))


def load_backend(kind: str, device: str, context_slots: int, steps: int, batch_size: int, model_id: str | None = None,
                 revision: str | None = None):
    """The backend of a foundation model, loaded once per process (weights from the Hugging Face Hub)."""
    model_id = model_id or FOUNDATION_MODELS[kind]
    key = (kind, model_id, revision, device, context_slots, steps)
    if key in _LOADED:
        return _LOADED[key]
    stale = [k for k in _LOADED if k[0] == kind and k[3] == device]  # another context length (script 43): one copy
    for k in stale:                                                   # of each model per process in memory
        del _LOADED[k]
    if stale and str(device).startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
    if key in _FAILED:
        raise RuntimeError(f"{kind} could not be loaded earlier in this process: {_FAILED[key]}")
    try:
        _LOADED[key] = _load(kind, model_id, device, context_slots, steps, batch_size, revision)
    except Exception as error:  # noqa: BLE001
        _FAILED[key] = f"{type(error).__name__}: {str(error).strip()}"[:300]
        raise
    return _LOADED[key]


def _load(kind: str, model_id: str, device: str, context_slots: int, steps: int, batch_size: int,
          revision: str | None = None):
    import torch

    if kind in ("chronos2", "chronos2_univariate"):
        from chronos import Chronos2Pipeline

        pipeline = Chronos2Pipeline.from_pretrained(model_id)
        pipeline.model = pipeline.model.to(device).eval()  # the pipeline follows the model's device
        backend = Chronos2Backend(pipeline, covariates=kind == "chronos2", batch_size=batch_size)
    elif kind == "timesfm":
        import timesfm

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id, torch_compile=False)
        model.model.device, model.model.device_count = torch.device(device), 1  # one device per worker process
        model.model.to(device)
        model.compile(timesfm.ForecastConfig(max_context=context_slots, max_horizon=steps,  # rounded up by compile
                                             normalize_inputs=True, per_core_batch_size=batch_size,
                                             use_continuous_quantile_head=True, force_flip_invariance=True,
                                             infer_is_positive=True, fix_quantile_crossing=True))
        backend = TimesFMBackend(model)
    elif kind == "ttm":
        from pathlib import Path

        if Path(model_id).is_dir():  # a saved checkpoint: its own context and prediction lengths
            from tsfm_public import TinyTimeMixerForPrediction

            model = TinyTimeMixerForPrediction.from_pretrained(model_id)
        else:  # the Hub card: the named checkpoint (D77), else the library's choice for the context and horizon
            from tsfm_public.toolkit.get_model import get_model

            options = {"model_revision": TTM_REVISIONS.get(revision, revision)} if revision else {"freq": "5min"}
            model = get_model(model_id, context_length=context_slots, prediction_length=steps, **options)
        model = model.to(device).eval()
        backend = TTMBackend(model, device, batch_size)
    else:
        raise ValueError(f"unknown foundation model {kind!r}; one of {sorted(FOUNDATION_MODELS)}")
    return backend


class FoundationModel(ForecastModel):
    mechanism = "context_only"

    def __init__(self, kind: str, context_hours: float = DEFAULT_CONTEXT_HOURS, device: str = "cpu",
                 batch_size: int = 256, model_id: str | None = None, backend=None, revision: str | None = None) -> None:
        if kind not in FOUNDATION_MODELS:
            raise ValueError(f"unknown foundation model {kind!r}; one of {sorted(FOUNDATION_MODELS)}")
        from ..compute import worker_device

        self.name = kind
        self.kind = kind
        self.context_hours = float(context_hours)
        self.device = worker_device(device)  # "cuda": the GPU of this worker process (D73)
        self.batch_size = scaled_batch(int(batch_size), self.context_hours)
        self.model_id = model_id
        self.revision = revision  # checkpoint (branch) on the Hub, for TTM (D77)
        self.backend = backend  # tests pass a backend built from random weights

    def fit(self, train: TrainData) -> FitLog:
        return FitLog(n_samples=0, details={"context_hours": self.context_hours, "start": "context",
                                            "device": self.device, "model_id": self.model_id or FOUNDATION_MODELS[self.kind],
                                            "revision": self.revision})

    def predict(self, samples: Samples) -> np.ndarray:
        if samples.context is None:
            raise ValueError("foundation models need samples built with context_h (Samples.context)")
        slots = int(round(self.context_hours * 60 / samples.step_min))
        if samples.context.shape[1] < slots:
            raise ValueError(f"the samples carry {samples.context.shape[1]} context slots, {slots} needed")
        from ..compute import gpu_slot

        steps = max(samples.horizon_steps)
        if not samples.n:
            return np.zeros((0, len(samples.horizons_min)), dtype=np.float32)
        with gpu_slot(self.device):
            backend = self.backend or load_backend(self.kind, self.device, slots, steps, self.batch_size,
                                                   self.model_id, self.revision)
            path = backend.forecast(samples, samples.context[:, -slots:], steps)
            if str(self.device).startswith("cuda"):  # hand the cached blocks back to the other workers of the GPU
                import torch

                torch.cuda.empty_cache()
        return clip(path[:, [k - 1 for k in samples.horizon_steps]])
