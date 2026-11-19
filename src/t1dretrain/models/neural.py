"""Networks from neuralforecast, trained on our own samples (HANDOFF 6.1, M09-M12 and M16; D60, D62, D64, D68, D74).

N-HiTS, TiDE, TSMixerx and PatchTST are the library's own modules
(``neuralforecast.models``); the LSTM is ours (D74, below). The training loop
is ours: the samples, origins, masks and validation days come from
``sim.samples`` so that every model sees the same data. Every network predicts
the whole path up to the longest horizon:

- networks that take historical inputs (N-HiTS, TiDE, TSMixerx, LSTM) read the
  glucose change, (glucose - glucose at the origin) / 50, and predict the
  change; the glucose level, (glucose - 140) / 50, is a historical input (D64:
  without it a network cannot learn that high values tend to fall and low
  values to rise), with IOB, COB, insulin activity and carbohydrate absorption
  (scaled); the time of day is a future input (known in advance);
- PatchTST takes no exogenous input, so it reads and predicts the level
  itself, (glucose - 140) / 50;
- missing inputs are carried forward and masked; the loss is the mean squared
  error over the path, ignoring slots without a valid reading; Adam; early
  stopping on the validation samples with the best weights kept, the starting
  weights included (step 0).

Three starting points (D62), chosen with ``start``:

- ``scratch``: a new network trained on the patient's window only;
- ``population``: the network pretrained on the pool group (script 11), used
  as it is (the patient's window is not used);
- ``finetune``: the pretrained network, trained further on the patient's
  window with the ``finetune`` schedule (smaller learning rate, few steps;
  ``train: head`` updates only the output layers, see ``HEADS``).

Two library networks do not work as direct multi-step forecasters and are
corrected (D74; ``NET_VERSION`` 2, files of version 1 are refused):

- TiDE: the residual block that gives each step's output ends with a layer
  normalisation over a single value, so its output is a constant and the
  forecast is only the linear skip from the past glucose changes (no level, no
  IOB/COB, no encoder). That last normalisation is switched off.
- LSTM: the library maps the forecast of step k to the hidden state of input
  step L - h + k, so the 5-min forecast ignored the last 115 minutes of
  glucose. Ours (``EncoderLSTM``) reads the whole window and forecasts the
  path from the last hidden state (with the future time of day) through an
  MLP head.

Batches are built from the samples on the fly, so that the population
samples (about a million windows) are not copied into network inputs at once.
The device is the GPU of the worker process (``cuda``, D73) or the CPU; if
CUDA fails the model is trained on the CPU.
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np

from ..compute import gpu_slot, worker_device
from ..sim.samples import Samples
from .base import EXOG, EXOG_SCALE, GLUCOSE_SCALE, FitLog, ForecastModel, TrainData, clip

# Library class, whether it takes historical/future inputs, and how its batches are laid out ("series": the
# multivariate layout of TSMixerx, [batch, channels, time, series])
NETWORKS = {"nhits": ("NHITS", True, "single"), "tide": ("TiDE", True, "single"), "tsmixerx": ("TSMixerx", True, "series"),
            "lstm": ("EncoderLSTM", True, "single"), "patchtst": ("PatchTST", False, "single")}
# Version of each network's construction, saved with the weights: a pretrained file of another version is refused
# (D74 changed TiDE and the LSTM)
NET_VERSION = {"nhits": 1, "tide": 2, "tsmixerx": 1, "lstm": 2, "patchtst": 1}
# Output layers trained by ``train: head`` (module-name prefixes; N-HiTS: the last layer of every block)
HEADS = {"tide": ("temporal_decoder", "global_skip"), "tsmixerx": ("out",), "lstm": ("mlp_decoder",),
         "patchtst": ("model.head",)}
DEFAULT_PARAMS = {"mlp_units": 256, "layers": 2, "dropout": 0.1, "pooling": "MaxPool1d", "exog": True, "level": True,
                  "learning_rate": 1e-3, "batch_size": 256, "max_steps": 1000, "val_every": 50, "patience": 5}
ARCHITECTURE = ("mlp_units", "layers", "dropout", "pooling", "exog", "level")
LEGACY = {"level": False, "layers": 2}  # architecture of networks saved before a parameter existed (release 0.17.0)
DEFAULT_FINETUNE = {"learning_rate": 1e-4, "batch_size": 256, "max_steps": 1000, "val_every": 25, "patience": 4,
                    "train": "all"}
SCHEDULE = ("learning_rate", "batch_size", "max_steps", "val_every", "patience")
HIST = ("iob", "cob", "insulin_activity", "carb_absorption")
GLUCOSE_CENTER = 140.0  # mg/dL: the level is (glucose - 140) / 50
STARTS = ("scratch", "population", "finetune")
# Once CUDA fails in a process (e.g. "unspecified launch failure"), its CUDA context is unusable: every later network
# of that process runs on the CPU (D72).
_CUDA_STATE = {"broken": False, "error": ""}


def is_cuda_error(error: BaseException) -> bool:
    text = f"{type(error).__name__} {error}".upper()
    return any(word in text for word in ("CUDA", "CUBLAS", "CUDNN", "ACCELERATOR"))


def usable_device(device: str) -> str:
    """The device a new network uses: ``cuda`` means the GPU of this worker process (D73); the CPU once CUDA has
    failed in this process."""
    device = worker_device(str(device))
    return "cpu" if device.startswith("cuda") and _CUDA_STATE["broken"] else device


def is_out_of_memory(error: BaseException) -> bool:
    return "OUT OF MEMORY" in f"{type(error).__name__} {error}".upper()


# Forecasts per forward pass when validating and predicting (no gradients; D73).
EVAL_BATCH = {"nhits": 2048, "tide": 512, "tsmixerx": 1024, "lstm": 1024, "patchtst": 1024}


def mark_cuda_broken(error: BaseException) -> None:
    _CUDA_STATE.update(broken=True, error=str(error).strip().splitlines()[0][:200] if str(error).strip() else "CUDA")


def _forward_fill(x: np.ndarray) -> np.ndarray:
    """Carry the last finite value forward along each row; leading gaps take the first finite value."""
    out = x.copy()
    for j in range(1, out.shape[1]):
        gap = ~np.isfinite(out[:, j])
        out[gap, j] = out[gap, j - 1]
    for j in range(out.shape[1] - 2, -1, -1):
        gap = ~np.isfinite(out[:, j])
        out[gap, j] = out[gap, j + 1]
    return out


def hist_channels(exog: bool, level: bool) -> list[str]:
    """Historical inputs of the network besides the glucose change: the glucose level and the exogenous series."""
    return (["level"] if level else []) + (list(HIST) if exog else [])


def network_inputs(s: Samples, rows: np.ndarray, path_steps: int, channels: list[str] | None = None,
                   absolute: bool = False) -> dict:
    """Network inputs of some samples (numpy): scaled glucose (change, or level if ``absolute``), mask,
    historical and future exogenous inputs, and the reference added back to the scaled outputs."""
    channels = hist_channels(True, True) if channels is None else channels
    x = np.asarray(s.inputs[rows], dtype=np.float32)
    g0 = x[:, -1:]
    mask = np.isfinite(x)
    filled = _forward_fill(x)
    reference = np.full_like(g0, GLUCOSE_CENTER) if absolute else g0
    y = np.nan_to_num((filled - reference) / GLUCOSE_SCALE, nan=0.0)
    columns = [np.nan_to_num((filled - GLUCOSE_CENTER) / GLUCOSE_SCALE, nan=0.0) if n == "level" else
               np.nan_to_num(np.asarray(s.exog[n][rows], dtype=np.float32) / EXOG_SCALE[n], nan=0.0) for n in channels]
    hist = np.stack(columns, axis=-1) if columns else np.zeros(x.shape + (0,), dtype=np.float32)
    window = x.shape[1]
    offsets = (np.arange(window + path_steps) - (window - 1)) * s.step_min * 60e9
    origin = np.asarray(s.origin_ts[rows]).astype("datetime64[ns]").astype(np.int64)
    hours = ((origin[:, None] + offsets[None, :]) % 86_400e9) / 3_600e9
    futr = np.stack([np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)], axis=-1)
    return {"y": y.astype(np.float32), "mask": mask.astype(np.float32), "hist": hist.astype(np.float32),
            "futr": futr.astype(np.float32), "reference": reference[:, 0].astype(np.float32), "g0": g0[:, 0]}


_LSTM_CLASS: dict = {}


def encoder_lstm(window: int, steps: int, n_hist: int, n_futr: int, units: int, layers: int, dropout: float):
    """The LSTM of M09 (D74): an LSTM encoder over the window (glucose change and historical inputs) and an MLP
    head (``mlp_decoder``) from the last hidden state and the future inputs to the whole path."""
    if "class" not in _LSTM_CLASS:
        import torch
        from torch import nn

        class EncoderLSTM(nn.Module):
            def __init__(self, window, steps, n_hist, n_futr, units, layers, dropout):
                super().__init__()
                self.window, self.steps, self.n_futr = window, steps, n_futr
                self.encoder = nn.LSTM(1 + n_hist, units, num_layers=layers, batch_first=True,
                                       dropout=dropout if layers > 1 else 0.0)
                self.mlp_decoder = nn.Sequential(nn.Linear(units + steps * n_futr, units), nn.ReLU(),
                                                 nn.Dropout(dropout), nn.Linear(units, steps))

            def forward(self, batch):
                x = batch["insample_y"]
                if batch["hist_exog"].shape[-1]:
                    x = torch.cat((x, batch["hist_exog"]), dim=-1)
                self.encoder.flatten_parameters()  # one block of weights again after loading a state (cuDNN)
                states, _ = self.encoder(x)
                state = states[:, -1]
                if self.n_futr:
                    state = torch.cat((state, batch["futr_exog"][:, -self.steps:].reshape(len(state), -1)), dim=1)
                return self.mlp_decoder(state)[..., None]  # [batch, steps, 1], as the library networks

        _LSTM_CLASS["class"] = EncoderLSTM
    return _LSTM_CLASS["class"](window, steps, n_hist, n_futr, units, layers, dropout)


def build_network(architecture: str, window: int, steps: int, params: dict, hist: list[str], futr: list[str],
                  seed: int):
    """The network for an architecture name of ``NETWORKS`` (library modules, corrected as described above)."""
    import neuralforecast.models as library
    from neuralforecast.losses.pytorch import MSE

    units, layers, dropout = int(params["mlp_units"]), int(params["layers"]), float(params["dropout"])
    common = dict(h=steps, input_size=window, loss=MSE(), random_seed=seed, scaler_type="identity")
    exog = dict(hist_exog_list=hist or None, futr_exog_list=futr or None)
    if architecture == "nhits":
        return library.NHITS(**common, **exog, n_pool_kernel_size=[max(1, window // 18), max(1, window // 36), 1],
                             pooling_mode=str(params["pooling"]),
                             n_freq_downsample=[max(1, steps // 4), max(1, steps // 8), 1],
                             mlp_units=3 * [[units, units]], dropout_prob_theta=dropout)
    if architecture == "tide":
        net = library.TiDE(**common, **exog, hidden_size=units, decoder_output_dim=16,
                           temporal_decoder_dim=max(32, units // 2), num_encoder_layers=layers,
                           num_decoder_layers=layers, dropout=dropout)
        net.temporal_decoder.layernorm = False  # a layer norm over one output value is a constant (D74)
        return net
    if architecture == "tsmixerx":
        return library.TSMixerx(**common, **exog, n_series=1, n_block=layers, ff_dim=units, dropout=dropout,
                                revin=False)
    if architecture == "lstm":
        import torch

        torch.manual_seed(seed)
        return encoder_lstm(window, steps, len(hist), len(futr), units, layers, dropout)
    if architecture == "patchtst":
        return library.PatchTST(**common, patch_len=12, stride=6, hidden_size=units, n_heads=4,
                                encoder_layers=layers, dropout=dropout, revin=False)
    raise ValueError(f"unknown network {architecture!r}; one of {sorted(NETWORKS)}")


class NeuralForecastModel(ForecastModel):
    mechanism = "global_finetune"
    exog = EXOG

    def __init__(self, architecture: str = "nhits", seed: int = 1, params: dict | None = None, device: str = "cpu",
                 start: str = "scratch", population: str | Path | None = None, finetune: dict | None = None) -> None:
        if architecture not in NETWORKS:
            raise ValueError(f"unknown network {architecture!r}; one of {sorted(NETWORKS)}")
        if start not in STARTS:
            raise ValueError(f"start must be one of {STARTS}")
        if start != "scratch" and population is None:
            raise ValueError(f"start={start!r} needs the file of a pretrained network")
        self.architecture = architecture
        self.seed = seed
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.finetune = {**DEFAULT_FINETUNE, **(finetune or {})}
        self.device = usable_device(device)
        self.start = start
        self.population = population
        self.net = None
        self.steps = 0
        self.window = 0
        self.name = architecture + {"scratch": "", "population": "_pop", "finetune": "_ft"}[start]

    # ------------------------------------------------------------------ network

    @property
    def takes_inputs(self) -> bool:
        """Whether the network reads historical and future inputs (else it reads the glucose level alone)."""
        return NETWORKS[self.architecture][1]

    def _channels(self) -> tuple[list[str], list[str]]:
        if not self.takes_inputs:
            return [], []
        exog = bool(self.params["exog"])
        return hist_channels(exog, bool(self.params["level"])), (["hour_sin", "hour_cos"] if exog else [])

    def _build(self, window: int, steps: int) -> None:
        import torch

        logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)
        logging.getLogger("lightning_fabric.utilities.seed").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*tensor.detach.*")
        torch.manual_seed(self.seed)
        self.window, self.steps = window, steps
        hist, futr = self._channels()
        self.net = build_network(self.architecture, window, steps, self.params, hist, futr, self.seed).to(self.device)

    def save(self, path: str | Path) -> None:
        import torch

        state = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        torch.save({"architecture": self.architecture, "version": NET_VERSION[self.architecture],
                    "params": self.params, "window": self.window, "steps": self.steps, "seed": self.seed,
                    "state": state}, Path(path))

    def load(self, path: str | Path) -> None:
        import torch

        saved = torch.load(Path(path), map_location="cpu", weights_only=False)
        if saved.get("architecture", "nhits") != self.architecture:
            raise ValueError(f"{path} holds a {saved.get('architecture')} network, not {self.architecture}")
        if int(saved.get("version", 1)) != NET_VERSION[self.architecture]:
            raise ValueError(f"{path} holds a {self.architecture} network of version {saved.get('version', 1)}; this "
                             f"release builds version {NET_VERSION[self.architecture]} (D74): run script 11 again "
                             f"with --families {self.architecture}")
        self.params = {**self.params, **{k: saved["params"].get(k, LEGACY.get(k, self.params[k])) for k in ARCHITECTURE}}
        self._build(int(saved["window"]), int(saved["steps"]))
        self.net.load_state_dict(saved["state"])

    def _tensors(self, s: Samples, rows: np.ndarray):
        import torch

        hist_names, futr_names = self._channels()
        a = network_inputs(s, rows, self.steps, hist_names, absolute=not self.takes_inputs)
        if not futr_names:
            a["futr"] = a["futr"][..., :0]
        d = self.device
        hist, futr = np.ascontiguousarray(a["hist"]), np.ascontiguousarray(a["futr"])
        if NETWORKS[self.architecture][2] == "series":  # [batch, channels, time, series]
            hist = np.ascontiguousarray(hist.transpose(0, 2, 1)[..., None])
            futr = np.ascontiguousarray(futr.transpose(0, 2, 1)[..., None])
            stat = torch.zeros((1, 0), device=d)
        else:
            stat = torch.zeros((len(rows), 0), device=d)
        batch = {"insample_y": torch.from_numpy(a["y"][:, :, None]).to(d),
                 "insample_mask": torch.from_numpy(a["mask"][:, :, None]).to(d),
                 "hist_exog": torch.from_numpy(hist).to(d), "futr_exog": torch.from_numpy(futr).to(d),
                 "stat_exog": stat}
        return batch, a["reference"]

    # ------------------------------------------------------------------ training

    def train_on(self, s: Samples, validation: np.ndarray | None, schedule: dict,
                 weight: np.ndarray | None = None) -> FitLog:
        """Train the network (built here if needed) on samples with a trajectory; early stopping on ``validation``.

        Without validation samples the network trains for ``max_steps`` steps (used for the final population
        network, whose number of steps comes from the tuning run).
        """
        start = None if self.net is None else {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        try:
            with gpu_slot(self.device):
                return self._train_on(s, validation, schedule, weight)
        except RuntimeError as error:  # CUDA failed: start the training again on the CPU, from the same weights
            if self.device == "cpu" or not is_cuda_error(error):
                raise
            if not is_out_of_memory(error):  # out of memory leaves CUDA usable; other failures do not
                mark_cuda_broken(error)
            self.device = "cpu"
            if start is None:
                self.net = None
            else:
                self._build(self.window, self.steps)
                self.net.load_state_dict(start)
            log = self._train_on(s, validation, schedule, weight)
            log.details["cuda_error"] = str(error).strip().splitlines()[0][:200]
            return log

    def _train_on(self, s: Samples, validation: np.ndarray | None, schedule: dict,
                  weight: np.ndarray | None = None) -> FitLog:
        import torch

        if s.trajectory is None:
            raise ValueError("networks are trained on the whole path: build the samples with trajectory=True")
        if self.net is None:
            self._build(s.inputs.shape[1], s.trajectory.shape[1])
        elif (s.inputs.shape[1], s.trajectory.shape[1]) != (self.window, self.steps):
            raise ValueError(f"the network expects windows of {self.window} and paths of {self.steps} slots")
        torch.manual_seed(self.seed)
        has_target = np.isfinite(np.asarray(s.trajectory)).any(axis=1)
        held = np.asarray(validation) if validation is not None else np.zeros(s.n, dtype=bool)
        fit_rows = np.flatnonzero(~held & has_target)
        val_rows = np.flatnonzero(held & has_target)
        if not len(fit_rows):
            raise ValueError("no training samples with a target")
        trained = self._trainable(str(schedule.get("train", "all")))
        optimizer = torch.optim.Adam(trained, lr=float(schedule["learning_rate"]))
        rng = np.random.default_rng(self.seed)
        batch = int(schedule["batch_size"])
        val_every, patience = int(schedule["val_every"]), int(schedule["patience"])

        def loss_on(rows: np.ndarray, reduce: bool = True):
            inputs, reference = self._tensors(s, rows)
            out = self.net(inputs)[..., 0]
            path = (np.asarray(s.trajectory[rows], dtype=np.float32) - reference[:, None]) / GLUCOSE_SCALE
            mask = np.isfinite(path).astype(np.float32)
            if weight is not None and reduce:  # weighted training loss (stage 2); validation stays unweighted
                mask = mask * np.asarray(weight, dtype=np.float32)[rows][:, None]
            m = torch.from_numpy(mask).to(self.device)
            y = torch.from_numpy(np.nan_to_num(path, nan=0.0)).to(self.device)
            se = ((out - y) ** 2 * m).sum()
            return se / m.sum().clamp(min=1) if reduce else (float(se), float(m.sum()))

        def validate() -> float:
            self.net.eval()
            total, count = 0.0, 0.0
            with torch.no_grad():
                for chunk in np.array_split(val_rows, max(1, len(val_rows) // EVAL_BATCH[self.architecture])):
                    se, n = loss_on(chunk, reduce=False)
                    total, count = total + se, count + n
            return total / max(count, 1.0)

        best, best_state, best_step, bad, step, curve = np.inf, None, 0, 0, 0, []
        if len(val_rows):  # the starting network competes too (a fine-tuning that only hurts is undone)
            best = validate()
            curve.append((0, round(best, 5)))
            best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        for step in range(1, int(schedule["max_steps"]) + 1):
            self.net.train()
            rows = np.sort(rng.choice(fit_rows, size=min(batch, len(fit_rows)), replace=False))
            optimizer.zero_grad()
            loss = loss_on(rows)
            loss.backward()
            optimizer.step()
            if len(val_rows) and step % val_every == 0:
                val = validate()
                curve.append((step, round(val, 5)))
                if val < best - 1e-6:
                    best, bad, best_step = val, 0, step
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return FitLog(n_samples=int(len(fit_rows)),
                      details={"steps": step, "best_step": best_step if len(val_rows) else step,
                               "best_val_mse": round(best, 5) if np.isfinite(best) else None, "device": self.device,
                               "start": self.start, "curve": curve})

    def head_parameters(self) -> list:
        """The output layers trained by ``train: head`` (N-HiTS: the projection to the coefficients of every
        block; the others: the modules named in ``HEADS``)."""
        if self.architecture == "nhits":
            return [p for block in self.net.blocks for p in block.layers[-1].parameters()]
        prefixes = HEADS[self.architecture]
        return [p for name, p in self.net.named_parameters() if name.startswith(prefixes)]

    def _trainable(self, which: str) -> list:
        """Parameters updated by the training: every one, or only the output layers (``head``; the rest of the
        network stays as pretrained)."""
        if which not in ("all", "head"):
            raise ValueError("train must be 'all' or 'head'")
        heads = {id(p) for p in self.head_parameters()} if which == "head" else set()
        if which == "head" and not heads:
            raise ValueError(f"no output layer found for {self.architecture}")
        for p in self.net.parameters():
            p.requires_grad_(which == "all" or id(p) in heads)
        return [p for p in self.net.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ fit / predict

    def fit(self, train: TrainData) -> FitLog:
        """Fit from the starting point; if CUDA fails on the way, the whole fit starts again on the CPU.
        ``compute_seconds`` leaves out the time spent waiting for a GPU turn (D73), so that it measures the cost of
        the fit itself (stage 2)."""
        try:
            with gpu_slot(self.device):
                begin = time.time()
                log = self._fit(train)
                log.details["compute_seconds"] = round(time.time() - begin, 3)
                return log
        except RuntimeError as error:
            if self.device == "cpu" or not is_cuda_error(error):
                raise
            if not is_out_of_memory(error):
                mark_cuda_broken(error)
            self.device, self.net = "cpu", None
            begin = time.time()
            log = self._fit(train)
            log.details["compute_seconds"] = round(time.time() - begin, 3)
            log.details["cuda_error"] = str(error).strip().splitlines()[0][:200]
            return log

    def _fit(self, train: TrainData) -> FitLog:
        if self.start == "scratch":
            self.net = None
            log = self.train_on(train.samples, train.validation, {k: self.params[k] for k in SCHEDULE},
                                train.sample_weight)
        else:
            self.load(self.population)
            if self.start == "population":
                return FitLog(n_samples=0, details={"start": self.start, "device": self.device})
            log = self.train_on(train.samples, train.validation, self.finetune, train.sample_weight)
        log.details.pop("curve", None)
        return log

    def continue_fit(self, train: TrainData, schedule: dict | None = None) -> FitLog:
        """Warm start (stage 3, D86): the current network trained further on the training window with the
        fine-tuning schedule (``schedule`` replaces it); undone if it only hurts the validation days."""
        if self.net is None:
            raise ValueError("no network to continue")
        with gpu_slot(self.device):
            begin = time.time()
            log = self.train_on(train.samples, train.validation, schedule or self.finetune, train.sample_weight)
            log.details.pop("curve", None)
            log.details.update(start="warm", compute_seconds=round(time.time() - begin, 3))
        return log

    def predict(self, samples: Samples) -> np.ndarray:
        """Forecasts in chunks of ``EVAL_BATCH``; if the GPU runs out of memory, smaller chunks, then the CPU."""
        size = EVAL_BATCH[self.architecture]
        while True:
            try:
                with gpu_slot(self.device):
                    return self._predict(samples, size)
            except RuntimeError as error:
                if self.device == "cpu" or not is_cuda_error(error):
                    raise
                if is_out_of_memory(error) and size > 16:
                    size //= 4
                    continue
                if not is_out_of_memory(error):
                    mark_cuda_broken(error)
                    raise  # a broken CUDA context cannot hand the weights to the CPU
                self.device = "cpu"
                self.net = self.net.to("cpu")

    def _predict(self, samples: Samples, size: int) -> np.ndarray:
        import torch

        steps = [k - 1 for k in samples.horizon_steps]
        self.net.eval()
        out, levels = [], []
        with torch.no_grad():
            for chunk in np.array_split(np.arange(samples.n), max(1, samples.n // size)):
                if len(chunk):
                    inputs, reference = self._tensors(samples, chunk)
                    out.append(self.net(inputs)[..., 0].cpu().numpy())
                    levels.append(reference)
        if not out:
            return np.zeros((0, len(steps)), dtype=np.float32)
        change, reference = np.concatenate(out), np.concatenate(levels)
        return clip(reference[:, None] + GLUCOSE_SCALE * change[:, steps])


class NHITSPersonal(NeuralForecastModel):
    """N-HiTS (M10), the first network of the project; same as ``NeuralForecastModel("nhits", ...)``."""

    def __init__(self, seed: int = 1, params: dict | None = None, device: str = "cpu", start: str = "scratch",
                 population: str | Path | None = None, finetune: dict | None = None) -> None:
        super().__init__("nhits", seed, params, device, start, population, finetune)
