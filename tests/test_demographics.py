"""Sex or gender of the patients (script 06, D108): normalisation, Loop's baseline survey and counts per cohort."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from t1dretrain.data.demographics import cohort_characteristics, female_indicator, normalize_sex
from t1dretrain.eval import paper_tables as pt
from t1dretrain.io import loop
from t1dretrain.io.common import PartWriter, TaskReport


def test_every_spelling_maps_to_three_categories():
    raw = pd.Series(["M", "F", "Cis Female ", "Cis Male", "Non-binary", "Woman", "prefer not to say", None, "x?"])
    assert normalize_sex(raw).tolist() == ["male", "female", "female", "male", "other", "female", pd.NA, pd.NA,
                                           pd.NA]
    assert normalize_sex(pd.Series(["1", "2", "3"]), codes={1: "male", 2: "female", 3: "non-binary"}).tolist() == [
        "male", "female", "other"]
    assert female_indicator(pd.Series(["F", "Cis Male", None])).tolist()[:2] == [1.0, 0.0]
    assert np.isnan(female_indicator(pd.Series([None])).iloc[0])


def test_loop_gender_comes_from_the_baseline_survey(tmp_path):
    (tmp_path / "PtRoster.txt").write_text(
        "PtID|EnrollDt|PtStatus|PtCohort|AgeAtEnrollment|LoopUseTimeAtEnroll|PtTimezoneOffset\n"
        "4|1/2/2018|Completed|B|20|More than 12 months|-5\n7|1/3/2018|Active|A|35|Not using Loop|-6\n"
        "9|1/4/2018|Active|A|50|Not using Loop|-6\n")
    (tmp_path / "LOOPDeviceUploads.txt").write_text("PtID|DeviceManufact|DeviceModel|DeviceType\n4|Dexcom|G6|cgm\n")
    (tmp_path / "Surveys.txt").write_text(
        "SubjectID|Period|PtCohort|ageAtBaseline|gender\n4|Baseline|B|20|1\n4|Month 6|B||\n7|Baseline|A|35|2\n")
    report = TaskReport()
    args = {"roster": tmp_path / "PtRoster.txt", "uploads": tmp_path / "LOOPDeviceUploads.txt", "final": None,
            "surveys": tmp_path / "Surveys.txt"}
    loop.read_patients(args, PartWriter(tmp_path / "parts", "patients"), report, {})
    sex = report.payload.set_index("patient_id")["sex"]
    assert sex["LOOP-4"] == "male" and sex["LOOP-7"] == "female" and pd.isna(sex["LOOP-9"])
    check = report.checks[0]
    assert check["surveys_in_roster"] == 2 and check["age_within_1_year"] == 2 and check["roster_with_gender"] == 2


def test_counts_per_cohort_and_core_group():
    cohorts = pd.DataFrame({"dataset": ["replace_bg"] * 3 + ["t1d_uom"] * 2,
                            "patient_id": ["R1", "R2", "R3", "U1", "U2"],
                            "cohort": ["core"] * 3 + ["open_insulin"] * 2,
                            "group": ["pool", "screen", "screen", None, None], "usable_days": [100] * 5})
    patients = pd.DataFrame({"dataset": ["replace_bg"] * 3 + ["t1d_uom"] * 2, "patient_id": ["R1", "R2", "R3", "U1", "U2"],
                             "sex": pd.Series(["female", "male", "female", pd.NA, pd.NA], dtype="string"),
                             "age": [30, 40, 50, np.nan, np.nan]})
    table = cohort_characteristics(cohorts, patients).set_index("key")
    assert table.loc["replace_bg/core", "women"] == 2 and table.loc["replace_bg/core", "women_pct"] == pytest.approx(
        200 / 3)
    assert table.loc["replace_bg/screen", "patients"] == 2 and table.loc["replace_bg/screen", "women_pct"] == 50
    assert np.isnan(table.loc["t1d_uom/open_insulin", "women_pct"])

    rows = [{"key": k, "dataset": "d", "cohort": "c", "therapy": "t", "sensor": "s", "inputs": "G", "role": "r"}
            for k in ("replace_bg/screen", "t1d_uom/open_insulin")]
    summary = pd.DataFrame({"dataset": ["t1d_uom"], "cohort": ["open_insulin"], "patients": [2],
                            "median_usable_days": [100]})
    cells = pt.cohort_cells(pt.cohort_table(rows, summary, cohorts, table.reset_index(), {}))
    assert cells["female_pct"].tolist() == ["50", pt.NOT_RECORDED]
    assert cells["age"].tolist() == ["45 (42--48)", pt.NOT_RECORDED]   # ages 40 and 50: median and quartiles
    pending = pt.cohort_cells(pt.cohort_table(rows, summary, cohorts, None, {}))
    assert pending["female_pct"].tolist() == [pt.PENDING, pt.PENDING]
    assert pending["age"].tolist() == [pt.PENDING, pt.PENDING]
