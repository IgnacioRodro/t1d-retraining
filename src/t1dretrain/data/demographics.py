"""Sex and gender of the patients, as each dataset records them (script 06; SAGER, D108).

The datasets record it differently: REPLACE-BG and DCLP3 a ``Gender`` column
with M or F; Loop a baseline survey question on gender with numeric codes
(1 male, 2 female, 3 non-binary; DataGlossary.rtf); BrisT1D a free-text answer
to "What would you describe your gender as?". T1D-UOM and HUPA-UCM publish no
sex or gender. ``normalize_sex`` maps every spelling to one of three categories
(female, male, other) and leaves the rest as not recorded; the paper reports
the share of women among the patients whose sex or gender is recorded.

``patients_table`` reads the static patients table of a dataset with its
loader (the ``patients`` task of ``plan``), without running the rest of the
harmonisation; ``cohort_characteristics`` counts the categories per cohort of
script 04 and per group of the core dataset.
"""

from __future__ import annotations

import importlib
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import LOADERS
from ..io.common import PartWriter, TaskReport

CATEGORIES = ("female", "male", "other")
FEMALE = {"f", "female", "woman", "women", "girl", "cis female", "cis woman", "cisgender female", "cisgender woman",
          "trans female", "trans woman", "transgender female", "transgender woman"}
MALE = {"m", "male", "man", "men", "boy", "cis male", "cis man", "cisgender male", "cisgender man", "trans male",
        "trans man", "transgender male", "transgender man"}
OTHER = {"non-binary", "nonbinary", "non binary", "genderqueer", "gender fluid", "genderfluid", "agender",
         "other", "intersex", "two-spirit"}
NOT_RECORDED = {"", "na", "n/a", "nan", "<na>", "none", "unknown", "prefer not to say", "prefer not to answer",
                "do not wish to answer", "not reported"}


def normalize_sex(values: pd.Series, codes: dict | None = None) -> pd.Series:
    """'female', 'male' or 'other' for every spelling a dataset uses (and numeric ``codes``, e.g. Loop's), <NA>
    when it is not recorded. An unknown spelling also counts as not recorded."""
    text = values.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    if codes:
        mapped = text.map({str(k).lower(): v for k, v in codes.items()})
        text = mapped.astype("string").fillna(text)

    def one(value) -> str | None:
        if value is None or pd.isna(value) or value in NOT_RECORDED:
            return None
        value = re.sub(r"[^a-z\- ]", "", value).strip()
        if value in FEMALE:
            return "female"
        if value in MALE:
            return "male"
        if value in OTHER:
            return "other"
        return None

    return pd.Series([one(v) for v in text], index=values.index, dtype="string")


def female_indicator(values: pd.Series) -> pd.Series:
    """1 for women, 0 for men and other genders, NaN when not recorded (the balance covariate of script 04)."""
    sex = normalize_sex(values)
    female = (sex == "female").fillna(False).to_numpy(dtype=bool)
    known = sex.isin(["male", "other"]).fillna(False).to_numpy(dtype=bool)
    return pd.Series(np.where(female, 1.0, np.where(known, 0.0, np.nan)), index=values.index)


def patients_table(dataset: str, root: Path, options: dict | None = None) -> tuple[pd.DataFrame, TaskReport]:
    """The static patients table of ``dataset`` as its loader reads it (the ``patients`` task of ``plan``), with the
    report of that task (counts and checks). Nothing else of the dataset is read."""
    module = importlib.import_module(LOADERS[dataset])
    tasks = [t for t in module.plan(root, dict(options or {})) if t.name == "patients"]
    if not tasks:
        raise ValueError(f"{dataset}: the loader has no patients task")
    report = TaskReport()
    with tempfile.TemporaryDirectory() as scratch:  # the patients task writes no rows; any part goes nowhere
        writer = PartWriter(Path(scratch), "patients")
        tasks[0].function(tasks[0].args, writer, report, dict(options or {}))
        writer.close()
    table = report.payload if isinstance(report.payload, pd.DataFrame) else pd.DataFrame(columns=["patient_id"])
    return table.drop_duplicates("patient_id"), report


def cohort_characteristics(cohorts: pd.DataFrame, patients: pd.DataFrame,
                           groups_of: tuple[str, str] = ("replace_bg", "core")) -> pd.DataFrame:
    """Per cohort of script 04 (key ``<dataset>/<cohort>``) and per group of the core cohort (key
    ``<dataset>/<group>``): patients, women, men, other genders, not recorded, the share of women among those
    recorded (%) and the age (median and quartiles). ``patients`` has ``dataset``, ``patient_id``, ``sex``
    (normalised) and ``age``."""
    merged = cohorts.merge(patients[["dataset", "patient_id", "sex", "age"]], on=["dataset", "patient_id"],
                           how="left")
    merged["key"] = merged["dataset"] + "/" + merged["cohort"]
    parts = [merged]
    dataset, cohort = groups_of
    core = merged[(merged["dataset"] == dataset) & (merged["cohort"] == cohort) & merged["group"].notna()
                  & (merged["group"].astype(str) != "")]
    if not core.empty:
        parts.append(core.assign(key=core["dataset"] + "/" + core["group"].astype(str)))
    rows = []
    for key, part in pd.concat(parts, ignore_index=True).groupby("key", sort=False):
        sex = part["sex"].astype("string")
        counts = {c: int((sex == c).sum()) for c in CATEGORIES}
        recorded = sum(counts.values())
        age = pd.to_numeric(part["age"], errors="coerce").dropna()
        rows.append({"key": key, "dataset": part["dataset"].iloc[0], "patients": int(part["patient_id"].nunique()),
                     "women": counts["female"], "men": counts["male"], "other_gender": counts["other"],
                     "not_recorded": int(len(part) - recorded),
                     "women_pct": 100 * counts["female"] / recorded if recorded else np.nan,
                     "age_median": float(age.median()) if len(age) else np.nan,
                     "age_q1": float(age.quantile(0.25)) if len(age) else np.nan,
                     "age_q3": float(age.quantile(0.75)) if len(age) else np.nan})
    return pd.DataFrame(rows)
