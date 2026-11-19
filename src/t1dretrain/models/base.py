"""Common contract of every forecasting model (HANDOFF section 6.3; first version, iteration 15).

A model is fitted on a training window of one patient (``TrainData``) and
predicts the glucose at every horizon for a set of samples
(``sim.samples.Samples``). Stage 1 will add ``adapt`` (update, re-estimate,
fine-tune or refit) on top of ``fit``; reference models need neither.

Predictions are clipped to the plausible glucose range of the common schema
(20-600 mg/dL); nothing else is done to them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..data.schema import PLAUSIBLE_GLUCOSE
from ..sim.samples import Samples


@dataclass
class TrainData:
    """What a model may learn from: samples of the training window and the raw series of those days."""

    samples: Samples
    series: np.ndarray            # glucose input on contiguous slots, NaN outside the training days
    segment: np.ndarray           # segment of every slot (clock breaks)
    step_min: float
    validation: np.ndarray | None = None  # samples held out for early stopping (the last days of the window)
    sample_weight: np.ndarray | None = None  # weight of every training sample (stage 2, exponential window; D80)
    series_weight: np.ndarray | None = None  # weight of every series slot (the same, for AR models)


@dataclass
class FitLog:
    seconds: float = 0.0
    n_samples: int = 0
    details: dict = field(default_factory=dict)


class ForecastModel:
    name: str = "model"
    mechanism: str = "reference"
    exog: tuple[str, ...] = ()
    seed: int = 0

    def fit(self, train: TrainData) -> FitLog:
        return FitLog()

    def predict(self, samples: Samples) -> np.ndarray:
        raise NotImplementedError

    def timed_fit(self, train: TrainData) -> FitLog:
        start = time.time()
        log = self.fit(train)
        log.seconds = round(time.time() - start, 4)
        return log


def clip(predictions: np.ndarray) -> np.ndarray:
    low, high = PLAUSIBLE_GLUCOSE
    return np.clip(predictions, low, high).astype(np.float32)


# Inputs of the learned models besides glucose (script 05 columns), and their scales for neural networks
EXOG = ("iob", "cob", "insulin_activity", "carb_absorption", "hour_sin", "hour_cos")
EXOG_SCALE = {"iob": 5.0, "cob": 50.0, "insulin_activity": 5.0, "carb_absorption": 50.0, "hour_sin": 1.0,
              "hour_cos": 1.0}
GLUCOSE_SCALE = 50.0  # mg/dL: neural inputs and outputs are (glucose - glucose at the origin) / 50

# D98 (script 73): the wearables a model can read at the origin (``features.build``: records that ended in the last
# 15 or 60 min, NaN where there are none), with the centre and scale of the linear models (ARX): a missing value
# counts as the centre.
WEARABLE_INPUTS = {"hr_15min": (80.0, 20.0), "steps_15min": (0.0, 500.0), "steps_60min": (0.0, 2000.0),
                   "kcal_15min": (0.0, 50.0), "asleep": (0.0, 1.0)}


def low_weights(targets: np.ndarray, low_loss: dict | None) -> np.ndarray | None:
    """D96 (script 71): the weight of every training target (samples x horizons), ``factor`` below ``threshold``
    mg/dL and 1 elsewhere (missing targets too, which are not used); None without ``low_loss``."""
    if not low_loss:
        return None
    t = np.asarray(targets, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.where(t < float(low_loss["threshold"]), float(low_loss["factor"]), 1.0)
