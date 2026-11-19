"""CGM cleaning rules of script 03 (src/t1dretrain/data/quality.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t1dretrain.data.quality import clean_cgm  # noqa: E402


def series(start: str, n: int, values, source: str = "dexcom_g6", step_min: float = 5.0, offset_s: float = 0.0):
    ts = pd.Timestamp(start) + pd.to_timedelta(np.arange(n) * step_min * 60 + offset_s, unit="s")
    values = np.broadcast_to(np.asarray(values, dtype=float), (n,))
    return pd.DataFrame({"ts": ts, "glucose_mgdl": values, "source": source, "record_type": "cgm"})


def check_accounting(stats: dict) -> None:
    removed = sum(stats[k] for k in ["removed_duplicates", "removed_isolated_conflicts", "removed_clock_overlap",
                                     "removed_cross_stream_duplicates", "removed_stream_disagreement",
                                     "removed_other_source_days"])
    assert stats["raw_auto_readings"] == stats["clean_readings"] + removed


def test_repeated_readings_are_dropped():
    base = series("2020-01-01", 288, np.linspace(100, 200, 288))
    repeat = base.iloc[::10].copy()
    repeat["ts"] += pd.Timedelta(seconds=20)
    clean, stats, diffs = clean_cgm(pd.concat([base, repeat]), "p1")
    assert len(clean) == 288 and stats["removed_duplicates"] == len(repeat)
    assert stats["clock_overlap_stretches"] == 0
    check_accounting(stats)


def test_clock_set_back_drops_the_overlapping_stretch():
    # 2 h of readings, then the clock goes back 1 h: the last hour appears twice with different values
    first = series("2020-10-25 00:00", 24, np.arange(24) * 5 + 100)
    second = series("2020-10-25 01:00", 24, np.arange(24) * 5 + 300, offset_s=40)
    clean, stats, diffs = clean_cgm(pd.concat([first, second]), "p1")
    assert stats["clock_overlap_stretches"] == 1 and stats["removed_clock_overlap"] == 12
    inside = (clean["ts"] >= pd.Timestamp("2020-10-25 01:00")) & (clean["ts"] <= pd.Timestamp("2020-10-25 01:56"))
    # no interleaved mixture: inside the overlap only the series that was running before the clock change is left
    assert inside.sum() == 12 and (clean.loc[inside, "glucose_mgdl"] < 300).all()
    assert len(clean) == 36
    # the clock jumps back after the overlap: the first reading after it starts a new segment
    assert clean.loc[clean["clock_break"], "ts"].tolist() == [pd.Timestamp("2020-10-25 02:00:40")]
    check_accounting(stats)


def test_time_shifted_copy_is_dropped():
    # 12 h of the series duplicated 3 h later with a 70-s offset (a copy with a shifted clock)
    values = 140 + 60 * np.sin(np.arange(576) / 25)
    base = series("2020-03-01", 576, values)
    copy = base.iloc[120:264].copy()
    copy["ts"] += pd.Timedelta(hours=3, seconds=70)
    clean, stats, diffs = clean_cgm(pd.concat([base, copy]), "p1")
    assert stats["clock_overlap_stretches"] >= 1
    assert len(clean) == len(base) and set(clean["ts"]) == set(base["ts"])
    check_accounting(stats)


def test_isolated_conflict_keeps_the_first_reading():
    base = series("2020-01-01", 100, 120.0)
    odd = base.iloc[[50]].copy()
    odd["ts"] += pd.Timedelta(seconds=60)
    odd["glucose_mgdl"] = 180.0
    clean, stats, diffs = clean_cgm(pd.concat([base, odd]), "p1")
    assert stats["removed_isolated_conflicts"] == 1 and len(clean) == 100
    assert (clean["glucose_mgdl"] == 120).all()
    check_accounting(stats)


def test_two_streams_of_one_sensor_fill_each_other():
    values = 100 + 30 * np.sin(np.arange(288) / 20)
    healthkit = series("2020-01-01", 288, values, source="dexcom_healthkit")
    direct = series("2020-01-01", 288, values, source="dexcom_direct", offset_s=10)
    healthkit = healthkit[(healthkit.index < 100) | (healthkit.index > 150)]  # a gap filled by the other stream
    clean, stats, diffs = clean_cgm(pd.concat([healthkit, direct.iloc[:200]]), "p1")
    assert len(clean) == 288
    assert stats["added_from_second_stream"] == 51 and stats["stream_disagreement_days"] == 0
    assert np.allclose(diffs["cross_stream"], 0)
    check_accounting(stats)


def test_shifted_streams_are_not_mixed():
    values = 100 + 60 * np.sin(np.arange(288) / 15)
    healthkit = series("2020-01-01", 288, values, source="dexcom_healthkit")
    shifted = series("2020-01-01", 288, np.roll(values, 12), source="dexcom_direct", offset_s=10)  # 1 h apart
    healthkit = healthkit[(healthkit.index < 100) | (healthkit.index > 150)]
    clean, stats, diffs = clean_cgm(pd.concat([healthkit, shifted.iloc[:200]]), "p1")
    assert stats["stream_disagreement_days"] == 1 and stats["added_from_second_stream"] == 0
    assert set(clean["source"]) == {"dexcom_healthkit"}
    check_accounting(stats)


def test_one_sensor_per_day():
    a = series("2020-01-01", 288 * 2, 120.0, source="dexcom_g6")
    b = series("2020-01-02 12:00", 288 * 2, 140.0, source="freestyle_libre", step_min=15)
    clean, stats, diffs = clean_cgm(pd.concat([a, b]), "p1")
    per_day = clean.groupby("day")["source"].nunique()
    assert (per_day == 1).all()
    check_accounting(stats)


def test_libre_scans_give_way_to_the_history():
    # 15-min history on a fixed phase plus frequent scans at odd times with different values (T1D-UOM)
    history = series("2020-01-01 00:01", 96 * 3, 100 + 40 * np.sin(np.arange(96 * 3) / 10), source="unspecified_cgm",
                     step_min=15)
    rng = np.random.default_rng(3)
    scan_times = history["ts"].iloc[100:140].to_numpy() + pd.to_timedelta(rng.uniform(2, 13, 40), unit="min")
    scans = pd.DataFrame({"ts": scan_times, "glucose_mgdl": rng.uniform(60, 250, 40), "source": "unspecified_cgm",
                          "record_type": "cgm"})
    clean, stats, diffs = clean_cgm(pd.concat([history, scans]), "p1")
    assert stats["clock_overlap_stretches"] == 0
    assert len(clean) == len(history) and set(clean["ts"]) == set(history["ts"])
    assert (clean["interval_min"] == 15).all()
    check_accounting(stats)
