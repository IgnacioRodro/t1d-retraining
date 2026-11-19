"""LightGBM, direct strategy (HANDOFF 6.1, M08; D60, D62).

One booster per horizon predicts the change from the glucose at the origin.
Features (all known at the origin): the glucose at the origin, every earlier
input of the window relative to it, IOB, COB, insulin activity and
carbohydrate absorption at the origin and 30 and 60 min before, and the time
of day (``exog: false`` keeps the glucose inputs only: the glucose-only twin of D76 and D90).
Missing inputs stay NaN (LightGBM routes them). Early stopping on the
validation samples. LightGBM is used directly rather than through mlforecast
because the samples (origins, windows, masks) come from ``sim.samples`` and
must be the same for every model.

Three starting points (D62), chosen with ``start``:

- ``scratch``: boosters trained on the patient's window only;
- ``population``: the boosters pretrained on the pool group (script 11), used
  as they are;
- ``finetune``: the pretrained boosters continued with new trees fitted on the
  patient's window (``init_model``), with the ``finetune`` parameters (smaller
  trees and learning rate); if the new trees do not lower the error on the
  validation days, the pretrained boosters are kept.

Stage 3 (D86) adds two ways to take in new data without going back to the
population model: ``continue_fit`` (warm start: new trees on the current
boosters) and ``refit_leaves`` (the trees are kept and their leaf values are
re-estimated with decay, LightGBM ``refit``).

Two options of the later analyses:

- ``low_loss`` ({"threshold", "factor"}; script 71, D96): every training
  target below ``threshold`` mg/dL weighs ``factor`` times, in the training
  loss and on the validation days (early stopping and the undo check), since
  the weighted loss is then what the model is asked to minimise;
- ``extra`` (grid columns, the wearables; script 73, D98): more inputs, read at
  the origin. The population boosters were trained without them, so a
  fine-tuning grows new boosters with every input on top of the population
  model's forecast (``init_score``): the same as continuing its boosting
  (``init_model``), which needs the same inputs, and identical to it when there
  are no extra inputs (``tests/test_model_options.py``). Only the restart can
  then refit (no warm start or leaf refit).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..sim.samples import Samples
from .base import EXOG, FitLog, ForecastModel, TrainData, clip, low_weights

DEFAULT_PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                  "max_rounds": 1000, "early_stopping_rounds": 50}
DEFAULT_FINETUNE = {"learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 50, "max_rounds": 300,
                    "early_stopping_rounds": 30}
ROUND_KEYS = ("max_rounds", "early_stopping_rounds")
OWN_KEYS = ROUND_KEYS + ("exog",)  # settings of ours, not LightGBM parameters
STARTS = ("scratch", "population", "finetune")
MIN_VALIDATION_ROWS = 50


def features(samples: Samples, rows: np.ndarray | slice = slice(None), exog: bool = True,
             extra: tuple[str, ...] = ()) -> np.ndarray:
    """The inputs of every sample; ``extra``: grid columns at the origin, appended last (D98)."""
    x = np.asarray(samples.inputs[rows], dtype=float)
    g0 = x[:, -1:]
    columns = [g0, x[:, :-1] - g0]
    added = [np.asarray(samples.exog[name][rows, -1], dtype=float)[:, None] for name in extra]
    if not exog:  # glucose only (D76)
        return np.hstack(columns + added).astype(np.float32)
    last = x.shape[1] - 1
    for name in ("iob", "cob", "insulin_activity", "carb_absorption"):
        e = samples.exog[name]
        for minutes in (0, 30, 60):
            k = min(last, int(round(minutes / samples.step_min)))
            columns.append(np.asarray(e[rows, last - k], dtype=float)[:, None])
    columns += [np.asarray(samples.exog["hour_sin"][rows, -1], dtype=float)[:, None],
                np.asarray(samples.exog["hour_cos"][rows, -1], dtype=float)[:, None]]
    return np.hstack(columns + added).astype(np.float32)


class LightGBMDirect(ForecastModel):
    name = "lightgbm"
    mechanism = "continue_training"
    exog = EXOG

    def __init__(self, seed: int = 1, params: dict | None = None, start: str = "scratch",
                 population: str | Path | None = None, finetune: dict | None = None, threads: int = 1,
                 low_loss: dict | None = None, extra: tuple[str, ...] = ()) -> None:
        if start not in STARTS:
            raise ValueError(f"start must be one of {STARTS}")
        if start != "scratch" and population is None:
            raise ValueError(f"start={start!r} needs the folder of pretrained boosters")
        self.seed = seed
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.finetune = {**DEFAULT_FINETUNE, **(finetune or {})}
        self.start = start
        self.population = population
        self.threads = int(threads)
        self.low_loss = dict(low_loss) if low_loss else None
        self.extra = tuple(extra)
        self.boosters: list = []
        self.personal: list = []  # with extra inputs: the boosters grown on top of the population ones (or None)
        self.extra_in_boosters = False  # the boosters themselves read the extra inputs (trained from scratch)
        self.horizons: list[int] = []
        if start != "scratch":
            self.name = f"lightgbm_{'pop' if start == 'population' else 'ft'}"

    # ------------------------------------------------------------------ training

    def train_on(self, s: Samples, validation: np.ndarray | None, params: dict, init: list | None = None,
                 weight: np.ndarray | None = None) -> FitLog:
        """Boosters for every horizon (continued from ``init`` when given); early stopping on ``validation``
        when it has at least 50 samples, else ``max_rounds`` rounds (an int, or one per horizon)."""
        import lightgbm as lgb

        exog = bool(self.params.get("exog", True))
        x = features(s, exog=exog, extra=self.extra)
        on_top = bool(init) and bool(self.extra)  # new boosters on the population model's forecast (D98)
        offsets = [b.predict(features(s, exog=exog), num_threads=self.threads) for b in init] if on_top else None
        g0 = np.asarray(s.inputs[:, -1], dtype=float)
        held = np.asarray(validation) if validation is not None else np.zeros(s.n, dtype=bool)
        low = low_weights(s.targets, self.low_loss)
        tree = {k: v for k, v in params.items() if k not in OWN_KEYS}
        tree.update({"seed": self.seed, "bagging_seed": self.seed, "feature_fraction_seed": self.seed,
                     "data_random_seed": self.seed, "deterministic": True, "num_threads": self.threads,
                     "verbose": -1})
        boosters, personal, rounds, added = [], [], [], []
        for k in range(len(s.horizons_min)):
            y = np.asarray(s.targets[:, k], dtype=float) - g0
            ok = np.isfinite(y)
            fit_rows, val_rows = ok & ~held, ok & held
            if not fit_rows.any():
                raise ValueError(f"no training samples with a target at {s.horizons_min[k]} min")
            w = np.ones(s.n) if weight is None else np.asarray(weight, dtype=float)
            if low is not None:
                w = w * low[:, k]
            w_val = low[val_rows, k] if low is not None else None
            score = offsets[k] if on_top else None
            data = lgb.Dataset(x[fit_rows], y[fit_rows], free_raw_data=True,
                               weight=None if weight is None and low is None else w[fit_rows],
                               init_score=None if score is None else score[fit_rows])
            callbacks, valid = [], []
            if val_rows.sum() >= MIN_VALIDATION_ROWS:
                valid = [lgb.Dataset(x[val_rows], y[val_rows], weight=w_val, reference=data,
                                     init_score=None if score is None else score[val_rows])]
                callbacks = [lgb.early_stopping(int(params["early_stopping_rounds"]), verbose=False)]
            start = init[k] if init else None
            before = start.current_iteration() if start is not None else 0
            limit = params["max_rounds"]
            limit = int(limit[k]) if isinstance(limit, (list, tuple)) else int(limit)
            booster = lgb.train(tree, data, num_boost_round=limit, valid_sets=valid, callbacks=callbacks,
                                init_model=None if on_top else start)
            best = int(booster.best_iteration or booster.current_iteration())
            if best < booster.current_iteration():  # keep the best iteration only (also when saved)
                booster = lgb.Booster(model_str=booster.model_to_string(num_iteration=best))
            if on_top:
                best += before
            if start is not None and valid:  # a fine-tuning that only hurts the validation days is undone
                xv, yv = x[val_rows], y[val_rows]
                kept = score[val_rows] if on_top else start.predict(xv)
                tuned = score[val_rows] + booster.predict(xv) if on_top else booster.predict(xv)
                wv = np.ones(len(yv)) if w_val is None else w_val
                if np.average((kept - yv) ** 2, weights=wv) <= np.average((tuned - yv) ** 2, weights=wv):
                    booster, best = (None if on_top else start), before
            boosters.append(start if on_top else booster)
            personal.append(booster if on_top else None)
            rounds.append(best)
            added.append(best - before)
        self.boosters, self.horizons = boosters, list(s.horizons_min)
        self.personal = personal if on_top else []
        self.extra_in_boosters = bool(self.extra) and not on_top
        details = {"rounds": rounds, "start": self.start}
        if init:
            details["added_rounds"] = added
        return FitLog(n_samples=int(s.n), details=details)

    def fit(self, train: TrainData) -> FitLog:
        if self.start == "scratch":
            return self.train_on(train.samples, train.validation, self.params, weight=train.sample_weight)
        self.load(self.population)
        if list(train.samples.horizons_min) != self.horizons:
            raise ValueError(f"pretrained boosters are for horizons {self.horizons}")
        if self.start == "population":
            return FitLog(n_samples=0, details={"start": self.start, "rounds": [b.current_iteration()
                                                                              for b in self.boosters]})
        return self.train_on(train.samples, train.validation, {**self.params, **self.finetune}, init=self.boosters,
                             weight=train.sample_weight)

    def continue_fit(self, train: TrainData) -> FitLog:
        """Warm start (stage 3, D86): new trees fitted on the training window are added to the current boosters,
        with the fine-tuning parameters and early stopping; undone if they do not help the validation days."""
        if not self.boosters:
            raise ValueError("no boosters to continue")
        if self.extra:
            raise NotImplementedError("a warm start with extra inputs (D98): only the restart refits")
        begin = time.time()
        log = self.train_on(train.samples, train.validation, {**self.params, **self.finetune}, init=self.boosters,
                            weight=train.sample_weight)
        log.details.update(start="warm", compute_seconds=round(time.time() - begin, 3))
        return log

    def refit_leaves(self, train: TrainData, decay: float = 0.9) -> FitLog:
        """The trees are kept and their leaf values re-estimated on the training window: new value = ``decay`` *
        old + (1 - ``decay``) * the value fitted on the window (LightGBM ``refit``; stage 3, D86)."""
        if not self.boosters:
            raise ValueError("no boosters to refit")
        if self.extra:
            raise NotImplementedError("a leaf refit with extra inputs (D98): only the restart refits")
        begin = time.time()
        s = train.samples
        x = features(s, exog=bool(self.params.get("exog", True)))
        g0 = np.asarray(s.inputs[:, -1], dtype=float)
        for k, booster in enumerate(self.boosters):
            y = np.asarray(s.targets[:, k], dtype=float) - g0
            ok = np.isfinite(y)
            if not ok.any():
                raise ValueError(f"no training samples with a target at {s.horizons_min[k]} min")
            self.boosters[k] = booster.refit(x[ok], y[ok], decay_rate=float(decay))
        return FitLog(n_samples=int(s.n), details={"start": "leaves", "decay": float(decay),
                                                   "compute_seconds": round(time.time() - begin, 3)})

    # ------------------------------------------------------------------ storage

    def save(self, folder: str | Path) -> None:
        """The boosters of every horizon (the population models of scripts 11 and 71)."""
        if self.personal:
            raise ValueError("boosters grown on top of a population model (extra inputs) are not saved")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for h, booster in zip(self.horizons, self.boosters):
            booster.save_model(str(folder / f"h{h}.txt"))
        (folder / "meta.json").write_text(json.dumps({"horizons_min": self.horizons, "params": self.params,
                                                      "seed": self.seed}, indent=1))

    def load(self, folder: str | Path) -> None:
        import lightgbm as lgb

        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        self.horizons = [int(h) for h in meta["horizons_min"]]
        self.params = {**self.params, **meta["params"]}
        self.boosters = [lgb.Booster(model_file=str(folder / f"h{h}.txt")) for h in self.horizons]
        self.personal, self.extra_in_boosters = [], False

    # ------------------------------------------------------------------ predict

    def predict(self, samples: Samples) -> np.ndarray:
        exog = bool(self.params.get("exog", True))
        g0 = np.asarray(samples.inputs[:, -1], dtype=float)
        if self.personal:  # the population model plus the boosters grown on top of it with the extra inputs
            base, x = features(samples, exog=exog), features(samples, exog=exog, extra=self.extra)
            change = np.column_stack([b.predict(base, num_threads=self.threads) +
                                      (p.predict(x, num_threads=self.threads) if p is not None else 0.0)
                                      for b, p in zip(self.boosters, self.personal)])
        else:
            x = features(samples, exog=exog, extra=self.extra if self.extra_in_boosters else ())
            change = np.column_stack([b.predict(x, num_threads=self.threads) for b in self.boosters])
        return clip(g0[:, None] + change)
