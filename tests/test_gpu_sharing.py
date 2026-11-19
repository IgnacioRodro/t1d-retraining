"""GPU sharing between worker processes (D73): one GPU per process, a limited number of processes per GPU at a
time, and a fall back to smaller chunks and then the CPU when the GPU runs out of memory."""

import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from t1dretrain import compute  # noqa: E402
from t1dretrain.eval.stage0_report import fit_outcomes  # noqa: E402

fcntl = pytest.importorskip("fcntl")


def test_gpu_slot_is_a_no_op_on_the_cpu_and_reentrant(tmp_path, monkeypatch):
    monkeypatch.setenv("T1D_GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("T1D_GPU_SLOTS", "1")
    with compute.gpu_slot("cpu") as outer:
        assert outer.handle is None
    with compute.gpu_slot("cuda:0") as outer:
        assert outer.handle is not None
        with compute.gpu_slot("cuda:0") as inner:  # the same process again: no second slot, no deadlock
            assert inner.handle is None
    assert compute._SLOT_STATE["depth"] == 0
    assert outer.handle is None  # released


def _hold(args):
    folder, path = args
    import os

    os.environ["T1D_GPU_LOCK_DIR"], os.environ["T1D_GPU_SLOTS"] = folder, "2"
    with compute.gpu_slot("cuda:1"):
        begin = time.time()
        time.sleep(0.4)
        end = time.time()
    with open(path, "a") as handle:
        handle.write(f"{begin} {end}\n")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # fork in a test process with threads
def test_gpu_slot_limits_the_processes_on_one_gpu(tmp_path):
    record = tmp_path / "times.txt"
    context = multiprocessing.get_context("fork")
    workers = [context.Process(target=_hold, args=((str(tmp_path / "locks"), str(record)),)) for _ in range(5)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(30)
    spans = np.array([[float(v) for v in line.split()] for line in record.read_text().splitlines()])
    assert len(spans) == 5
    for t in np.linspace(spans.min(), spans.max(), 200):  # never more than two processes inside at the same time
        assert ((spans[:, 0] < t) & (spans[:, 1] > t)).sum() <= 2


def test_workers_are_spread_over_the_gpus(tmp_path, monkeypatch):
    monkeypatch.setenv("T1D_GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(compute.os, "getppid", lambda: 4242)
    assert [compute._least_used_gpu(2) for _ in range(5)] == [0, 1, 0, 1, 0]
    assert compute.worker_device("cpu") == "cpu" and compute.worker_device("cuda:1") == "cuda:1"


def test_share_gpus_takes_the_setting_unless_the_environment_has_one(tmp_path, monkeypatch):
    monkeypatch.setenv("T1D_GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.delenv("T1D_GPU_SLOTS", raising=False)
    assert compute.share_gpus({"compute": {"gpu_slots_per_gpu": 3}}) == 3
    monkeypatch.setenv("T1D_GPU_SLOTS", "6")
    assert compute.share_gpus({"compute": {"gpu_slots_per_gpu": 3}}) == 6


def test_out_of_memory_uses_smaller_chunks_then_the_cpu(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("neuralforecast")
    from test_learned import pool_data

    from t1dretrain.models import neural
    from t1dretrain.models.base import TrainData

    monkeypatch.setenv("T1D_GPU_LOCK_DIR", str(tmp_path))
    s, held = pool_data(5, days=3)
    model = neural.NeuralForecastModel("lstm", 1, {"mlp_units": 16, "max_steps": 5, "val_every": 5})
    model.fit(TrainData(s, np.zeros(0), np.zeros(0), s.step_min, validation=held))
    expected = model.predict(s)
    original, sizes = neural.NeuralForecastModel._predict, []

    def tight(self, samples, size):  # the GPU never has room: every chunk size fails until the CPU takes over
        sizes.append((self.device, size))
        if self.device != "cpu":
            raise RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB")
        return original(self, samples, size)

    monkeypatch.setattr(neural.NeuralForecastModel, "_predict", tight)
    monkeypatch.setitem(neural._CUDA_STATE, "broken", False)
    model.device = "cuda:0"
    got = model.predict(s)
    tried, size = [], neural.EVAL_BATCH["lstm"]
    while True:
        tried.append(size)
        if size <= 16:
            break
        size //= 4
    assert [size for device, size in sizes if device != "cpu"] == tried
    assert model.device == "cpu" and not neural._CUDA_STATE["broken"]  # out of memory does not break CUDA
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-4)


def test_fit_outcomes_counts_each_cause():
    log = pd.DataFrame({"model": ["a", "a", "a", "b"],
                        "error": [None, "ValueError: no training samples in the window", "RuntimeError: CUDA", None],
                        "device": ["cuda:0", None, None, "cpu"], "cuda_error": [None, None, None, "CUDA error"]})
    table = fit_outcomes(log).set_index("model")
    assert table.loc["a", ["fits", "ok", "on_gpu", "data_failure", "other_failure"]].tolist() == [3, 1, 1, 1, 1]
    assert table.loc["b", ["ok", "cpu_fallback", "other_failure"]].tolist() == [1, 1, 0]
