"""Loader rules that decide units from the data itself."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t1dretrain.io.replace_bg import carb_input_mode  # noqa: E402


def test_replace_bg_carb_mode_from_the_wizard_arithmetic():
    carbs = pd.Series([45.0, 3.0, 2.0, 10.0, 30.0, 4.0])
    ratio = pd.Series([9.0, 1.5, 1.0, 10.0, None, 10.0])   # g/U, U/exchange, ambiguous (1), g/U, missing, g/U
    recommended = pd.Series([5.0, 4.5, 2.0, 1.0, 3.0, 6.0])  # last: 4 exchanges = 60 g at 10 g/U -> 6 U
    mode = carb_input_mode(carbs, ratio, recommended)
    assert mode.where(mode.notna(), "undecided").tolist() == ["grams", "exchanges", "undecided", "grams", "undecided",
                                                             "exchanges"]
