"""Clarke Error Grid zones, metrics and the saved histogram (rule R6)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t1dretrain.eval.metrics import (ClarkeAccumulator, clarke_histogram, clarke_zones, metrics_table,  # noqa: E402
                                     point_metrics, zone_table)

# (reference, forecast, zone) worked out by hand from the zone definitions
KNOWN = [
    (100, 100, "A"), (50, 60, "A"), (65, 75, "A"), (100, 80, "A"), (100, 120, "A"), (300, 250, "A"),
    (100, 125, "B"), (100, 60, "B"), (300, 200, "B"), (200, 140, "B"), (60, 40, "A"),
    (100, 220, "C"), (150, 25, "C"), (170, 100, "B"), (180, 70, "E"),
    (250, 100, "D"), (50, 100, "D"), (65, 85, "D"), (300, 180, "D"),
    (200, 70, "E"), (60, 200, "E"), (400, 20, "E"), (30, 300, "E"),
]


def test_known_points():
    reference = np.array([k[0] for k in KNOWN], dtype=float)
    forecast = np.array([k[1] for k in KNOWN], dtype=float)
    expected = [k[2] for k in KNOWN]
    assert list(clarke_zones(reference, forecast)) == expected


def test_metrics_and_shares():
    reference = np.array([100, 100, 200, 60, np.nan])
    forecast = np.array([110, 90, 70, 200, 100])
    m = point_metrics(reference, forecast)
    assert m["n"] == 4
    errors = np.array([10, -10, -130, 140])
    assert np.isclose(m["rmse"], np.sqrt(np.mean(errors ** 2)))
    assert np.isclose(m["mae"], np.mean(np.abs(errors)))
    assert np.isclose(m["bias"], errors.mean())
    assert np.isclose(m["mard_pct"], 100 * np.mean(np.abs(errors) / np.array([100, 100, 200, 60])))
    assert np.isclose(m["clarke_A_pct"], 50) and np.isclose(m["clarke_E_pct"], 50)
    assert np.isclose(m["clarke_AB_pct"], 50)


def test_histogram_reproduces_exact_zone_counts():
    rng = np.random.default_rng(1)
    reference = rng.uniform(40, 400, 50000)
    forecast = reference + rng.normal(0, 40, len(reference))
    hist = clarke_histogram(reference, forecast)
    assert hist["count"].sum() == len(reference)
    exact = pd.Series(clarke_zones(reference, forecast)).value_counts()
    from_hist = hist.groupby("zone")["count"].sum()
    assert (from_hist.reindex(exact.index) == exact).all()
    shares = zone_table(hist.assign(panel="x"), ["panel"]).iloc[0]
    assert np.isclose(shares["clarke_A_pct"], point_metrics(reference, forecast)["clarke_A_pct"])


def test_accumulator_merges_workers():
    rng = np.random.default_rng(2)
    reference = rng.uniform(40, 400, 30000)
    forecast = reference * rng.normal(1, 0.15, len(reference))
    one = ClarkeAccumulator(["model", "horizon_min"], sample_size=1000, stream=1)
    two = ClarkeAccumulator(["model", "horizon_min"], sample_size=1000, stream=2)
    one.add(("arima", 30), reference[:10000], forecast[:10000])
    two.add(("arima", 30), reference[10000:], forecast[10000:])
    two.add(("lgbm", 30), reference, forecast)
    one.merge(two)
    zones = one.zones().set_index("model")
    assert zones.loc["arima", "n"] == 30000 and zones.loc["lgbm", "n"] == 30000
    assert np.isclose(zones.loc["arima", "clarke_A_pct"], point_metrics(reference, forecast)["clarke_A_pct"])
    sample = one.sample()
    assert (sample.groupby("model").size() == 1000).all()


def test_metrics_table_groups():
    frame = pd.DataFrame({"model": ["a", "a", "b"], "reference": [100, 200, 100], "forecast": [100, 220, 150]})
    table = metrics_table(frame, ["model"])
    assert list(table["model"]) == ["a", "b"] and list(table["n"]) == [2, 1]
