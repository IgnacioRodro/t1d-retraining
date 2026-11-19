"""Helpers shared by the dataset loaders (script 02).

Reading: files are located inside a dataset folder by name (one level of extra
nesting after unzipping does not matter), and their encoding, delimiter and
header line are detected rather than assumed. Column names are matched without
regard to case, because the same field is spelt ``PtID`` in one table and
``PtId`` in the next.

Writing: loaders do not build whole datasets in memory. A ``PartWriter`` keeps
the rows read by one task per patient and table and flushes them to small
parquet "part" files; script 02 later merges the parts of each patient in
parallel (see ``t1dretrain.data.pipeline``).
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.schema import MGDL_PER_MMOL, TABLES
from ..data.store import safe_id
from .sniff import detect_delimiter, detect_encoding, detect_header_offset, read_text_sample

TIDEPOOL_MGDL_PER_MMOL = 18.01559  # factor Tidepool used to convert mg/dL to mmol/L (Loop exports)


# ----------------------------------------------------------------------------- locating files


def find_files(root: Path, pattern: str) -> list[Path]:
    """Files under ``root`` whose name matches the glob ``pattern`` (case-insensitive), sorted."""
    regex = re.compile(_glob_to_regex(pattern), re.IGNORECASE)
    return sorted(p for p in root.rglob("*") if p.is_file() and regex.fullmatch(p.name))


def find_one(root: Path, pattern: str, required: bool = True) -> Path | None:
    """The shallowest file under ``root`` whose name matches ``pattern``."""
    matches = sorted(find_files(root, pattern), key=lambda p: (len(p.parts), str(p)))
    if not matches:
        if required:
            raise FileNotFoundError(f"No file matching '{pattern}' under {root}")
        return None
    return matches[0]


def _glob_to_regex(pattern: str) -> str:
    return "".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern)


# ----------------------------------------------------------------------------- reading tables


@dataclass
class TableFormat:
    encoding: str
    delimiter: str
    header_offset: int
    columns: list[str]


def sniff_table(path: Path, delimiter: str | None = None) -> TableFormat:
    """Encoding, delimiter, header line and column names of a delimited text file."""
    encoding = detect_encoding(path)
    sample = read_text_sample(path, encoding)
    sample = sample.replace("\r\n", "\n").replace("\r", "\n")
    delimiter = delimiter or detect_delimiter(sample) or ","
    offset = detect_header_offset(sample, delimiter)
    header_line = sample.split("\n")[offset]
    columns = next(csv.reader([header_line], delimiter=delimiter), [])
    columns = [str(c).strip().lstrip("﻿") for c in columns]
    return TableFormat(encoding, delimiter, offset, columns)


def resolve_columns(available: Sequence[str], wanted: Sequence[str], required: bool = True) -> dict[str, str]:
    """Map each wanted name to the actual column name, ignoring case and surrounding spaces."""
    lookup = {str(c).strip().lower(): c for c in available}
    mapping = {}
    for name in wanted:
        actual = lookup.get(name.strip().lower())
        if actual is None:
            if required:
                raise KeyError(f"Column '{name}' not found; available: {list(available)}")
            continue
        mapping[name] = actual
    return mapping


def read_table(
    path: Path,
    columns: Sequence[str] | None = None,
    optional: Sequence[str] = (),
    chunksize: int | None = None,
    max_chunks: int | None = None,
    delimiter: str | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Read a delimited file as strings, with detected format and case-insensitive column names.

    ``columns`` must exist, ``optional`` columns are read when present (and
    created empty otherwise). The returned frames use the spelling given in
    ``columns``/``optional``. With ``chunksize`` an iterator of frames is
    returned (at most ``max_chunks`` of them).
    """
    fmt = sniff_table(path, delimiter)
    mapping: dict[str, str] = {}
    if columns is not None:
        mapping.update(resolve_columns(fmt.columns, columns, required=True))
    mapping.update(resolve_columns(fmt.columns, optional, required=False))
    actual_to_wanted = {actual: wanted for wanted, actual in mapping.items()}
    # pandas compares usecols with the raw header, which may keep a BOM on the first name
    usecols = None if columns is None and not optional else (lambda c: str(c).strip().lstrip("﻿") in actual_to_wanted)
    reader = pd.read_csv(
        path,
        sep=fmt.delimiter,
        encoding=fmt.encoding,
        skiprows=fmt.header_offset,
        usecols=usecols,
        dtype=str,
        keep_default_na=True,
        na_values=["", "NA", "NULL", "null", "NaN"],
        chunksize=chunksize,
        low_memory=False,
        encoding_errors="replace",
    )

    def tidy(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.rename(columns=lambda c: actual_to_wanted.get(str(c).strip().lstrip("﻿"), str(c).strip()))
        for name in list(columns or []) + list(optional):
            if name not in frame:
                frame[name] = pd.NA
        return frame

    if chunksize is None:
        return tidy(reader)

    def iterate() -> Iterator[pd.DataFrame]:
        with reader as chunks:
            for index, chunk in enumerate(chunks):
                if max_chunks is not None and index >= max_chunks:
                    break
                yield tidy(chunk)

    return iterate()


def read_text(path: Path) -> str:
    """Whole text file decoded with the detected encoding and with ``\\n`` line ends only."""
    encoding = detect_encoding(path)
    text = path.read_bytes().decode(encoding, errors="replace").lstrip("﻿")
    return text.replace("\r\n", "\n").replace("\r", "\n")


# ----------------------------------------------------------------------------- values


def to_number(values: pd.Series) -> pd.Series:
    """Numbers from strings; accepts a decimal comma."""
    if values.dtype != object and not pd.api.types.is_string_dtype(values):
        return pd.to_numeric(values, errors="coerce").astype("float64")
    cleaned = values.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")  # plain floats: NaN, never pd.NA


def mmol_to_mgdl(values: pd.Series, factor: float = MGDL_PER_MMOL, decimals: int | None = None) -> pd.Series:
    out = to_number(values) * factor
    return out.round(decimals) if decimals is not None else out


def censored_glucose(values: pd.Series, low_tokens: Sequence[str], high_tokens: Sequence[str],
                     low_value: float, high_value: float) -> tuple[pd.Series, int, int]:
    """Numbers from a glucose column where 'low'/'high' strings stand for out-of-range readings.

    Returns the numeric series and the number of low and high replacements.
    """
    text = values.astype("string").str.strip().str.lower()
    low = text.isin([t.lower() for t in low_tokens])
    high = text.isin([t.lower() for t in high_tokens])
    numbers = to_number(values)
    numbers = numbers.mask(low, low_value).mask(high, high_value)
    return numbers, int(low.sum()), int(high.sum())


def parse_datetimes(values: pd.Series, formats: Sequence[str]) -> tuple[pd.Series, str]:
    """Parse with the candidate format that reads the most values; returns (timestamps, format).

    Order matters for ties: put the format the documentation or the data
    supports first (for example day-first when days above 12 are seen).
    """
    text = values.astype("string").str.strip()
    best, best_format, best_count = None, "none", -1
    for fmt in formats:
        if fmt == "ISO8601":
            parsed = pd.to_datetime(text, format="ISO8601", errors="coerce")
        else:
            parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        count = int(parsed.notna().sum())
        if count > best_count:
            best, best_format, best_count = parsed, fmt, count
        if count == int(text.notna().sum()):
            break
    return best, best_format


def numeric_timedelta(values: pd.Series, unit: str, low: float, high: float, report: "TaskReport | None" = None,
                      source: str = "", column: str = "") -> pd.Series:
    """Timedeltas from numbers in ``unit``; values outside [low, high] (or not finite) become NaT and are counted.

    pandas multiplies the numbers by the unit length in nanoseconds; a corrupt
    value (say 1e20 minutes) overflows and turns into a silent garbage
    timestamp. Every numeric-to-time conversion of the loaders goes through
    this guard (iteration 10: numpy overflow warnings in the read phase).
    """
    numbers = to_number(values) if not pd.api.types.is_float_dtype(values) else values.astype("float64")
    bad = numbers.notna() & (~np.isfinite(numbers) | (numbers < low) | (numbers > high))
    if report is not None and bad.any():
        report.tally(source, column, f"outside [{low:g}, {high:g}] {unit} (ignored)", int(bad.sum()))
    return pd.to_timedelta(numbers.where(~bad), unit=unit)


# ----------------------------------------------------------------------------- loop status from device names

CLOSED_LOOP_KEYWORDS = ["780g", "control-iq", "control iq", "omnipod 5", "camaps", "ilet", "androidaps", "aaps",
                        "openaps", "diabeloop", "dbl-4", "medtronic 770g", "loop"]
PROBABLE_CLOSED_KEYWORDS = ["t:slim", "tslim", "t slim", "670g", "ypsopump", "dana", "mylife"]
OPEN_KEYWORDS = ["pen", "640g", "veo", "paradigm", "omnipod dash", "omnipod eros", "insight", "combo", "accu-chek"]


def loop_status(devices: Sequence[str]) -> str:
    """'closed', 'closed_probable', 'open' or 'unknown' from insulin-device names."""
    names = " | ".join(str(d).lower() for d in devices if d and str(d) != "nan")
    if not names:
        return "unknown"
    if any(k in names for k in CLOSED_LOOP_KEYWORDS):
        return "closed"
    if any(k in names for k in PROBABLE_CLOSED_KEYWORDS):
        return "closed_probable"
    if any(k in names for k in OPEN_KEYWORDS):
        return "open"
    return "unknown"


# ----------------------------------------------------------------------------- task reports


@dataclass
class TaskReport:
    """What a loader task tells the script, besides the rows it writes."""

    rows_read: int = 0
    value_counts: Counter = field(default_factory=Counter)   # (source, column, value) -> n
    meta: dict[str, dict[str, Any]] = field(default_factory=lambda: defaultdict(dict))  # patient -> key -> value
    devices: list[dict[str, Any]] = field(default_factory=list)  # device periods
    notes: list[str] = field(default_factory=list)
    payload: Any = None  # e.g. the static patients table
    checks: list[dict[str, Any]] = field(default_factory=list)  # rows of loader-specific check tables

    def add_check(self, name: str, rows: pd.DataFrame) -> None:
        """Rows for the check table ``checks_<name>.csv`` written by the script."""
        self.checks += [{"check": name, **row} for row in rows.to_dict("records")]

    def count(self, source: str, column: str, values: pd.Series, top: int = 50) -> None:
        """Record the frequency of the values of a categorical source column."""
        counts = values.astype("string").fillna("<NA>").value_counts()
        for value, n in counts.head(top).items():
            self.value_counts[(source, column, str(value)[:80])] += int(n)
        if len(counts) > top:
            self.value_counts[(source, column, "<other values>")] += int(counts.iloc[top:].sum())

    def tally(self, source: str, column: str, value: str, n: int) -> None:
        if n:
            self.value_counts[(source, column, value)] += int(n)

    def set_meta(self, patient_id: str, **values: Any) -> None:
        self.meta[patient_id].update({k: v for k, v in values.items() if v is not None})

    def add_meta_count(self, patient_id: str, key: str, n: int) -> None:
        """Counters merged by summation across tasks (keys end with '_n')."""
        self.meta[patient_id][key] = self.meta[patient_id].get(key, 0) + int(n)


# ----------------------------------------------------------------------------- part writer


class PartWriter:
    """Buffers rows per (table, patient) and writes them as parquet parts.

    Parts live in ``<parts_root>/<table>/<patient>/<task>-<k>.parquet``; the
    buffer is flushed whenever it holds more than ``flush_rows`` rows, so a task
    reading a multi-GB file never keeps it all in memory.
    """

    def __init__(self, parts_root: Path, task_name: str, flush_rows: int = 3_000_000) -> None:
        self.root = parts_root
        self.task = safe_id(task_name)
        self.flush_rows = flush_rows
        self._buffer: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
        self._buffered = 0
        self._serial = 0
        self.rows_written: Counter = Counter()

    def add(self, table: str, frame: pd.DataFrame) -> None:
        """Add rows (must contain ``patient_id``) for one table; split by patient."""
        if frame is None or frame.empty:
            return
        columns = list(TABLES.get(table, {}))  # auxiliary tables (e.g. Loop time-zone offsets) keep their order
        extra = [c for c in frame.columns if c not in columns]
        frame = frame[[c for c in columns if c in frame.columns] + extra]
        for patient_id, part in frame.groupby("patient_id", sort=False):
            self._buffer[(table, str(patient_id))].append(part)
            self._buffered += len(part)
        if self._buffered >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        for (table, patient_id), parts in self._buffer.items():
            frame = pd.concat(parts, ignore_index=True)
            target = self.root / table / safe_id(patient_id) / f"{self.task}-{self._serial:04d}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(target, index=False)
            self.rows_written[table] += len(frame)
        self._buffer.clear()
        self._buffered = 0
        self._serial += 1

    def close(self) -> None:
        if self._buffered:
            self.flush()


DURATION_MS_THRESHOLD = 10_000  # a median duration above this looks like milliseconds, below it like minutes


def duration_minutes(duration: pd.Series, upload_id: pd.Series, source: pd.Series, report: "TaskReport",
                     table: str) -> pd.Series:
    """Durations in minutes for JAEB tables whose unit depends on the upload source.

    The glossaries say Tidepool uploads give milliseconds and Diasend uploads
    minutes; that documented rule is applied (decision D33, revised in
    iteration 8). Magnitude is only a check: basal segments and extended
    boluses last minutes to hours, i.e. ~1e5-1e8 in ms and ~5-1500 in minutes.
    If a whole documented group contradicts its unit (median on the wrong side
    of 10 000), that group is switched and the override is reported. Uploads
    whose own median disagrees are only counted: single uploads with a few
    very short segments (seconds) fooled a per-upload rule in iteration 7.
    """
    values = to_number(duration)
    documented_minutes = source.astype("string").str.lower().str.contains("diasend", na=False).to_numpy()
    in_ms = ~documented_minutes
    positive = values.where(values > 0)
    for group, label in [(~documented_minutes, "ms (Tidepool/other)"), (documented_minutes, "min (Diasend)")]:
        if not group.any() or positive[group].notna().sum() == 0:
            continue
        median = float(positive[group].median())
        looks_ms = median >= DURATION_MS_THRESHOLD
        if looks_ms != (label.startswith("ms")):
            in_ms = np.where(group, looks_ms, in_ms)
            report.tally(table, "duration unit", f"documented {label} overridden by magnitude", int(group.sum()))
        else:
            report.tally(table, "duration unit", f"documented {label} confirmed by magnitude",
                         int((group & values.notna().to_numpy()).sum()))
    upload = upload_id.astype("string").str.strip().fillna("<none>")
    upload_looks_ms = upload.map(positive.groupby(upload).median() >= DURATION_MS_THRESHOLD).fillna(True).to_numpy()
    disagree = (upload_looks_ms != in_ms) & values.notna().to_numpy()
    report.tally(table, "duration unit", "rows in uploads whose own median disagrees (kept documented unit)",
                 int(disagree.sum()))
    return pd.Series(np.where(in_ms, values / 60000.0, values), index=values.index)


def daily_timedelta(days: pd.Series, clock: pd.Series, report: "TaskReport | None" = None,
                    source: str = "") -> pd.Series:
    """Offset from a base date: whole days (within +-10 years) plus a 'HH:MM:SS' clock time."""
    clock_delta = pd.to_timedelta(clock.astype("string").str.strip(), errors="coerce")
    return numeric_timedelta(days, "D", -3650, 3650, report, source, "days from enrolment") + clock_delta


def first_valid(*series: pd.Series) -> pd.Series:
    """Element-wise first non-missing value among aligned series."""
    out = series[0].copy()
    for other in series[1:]:
        out = out.where(out.notna(), other)
    return out


def finite_positive(values: pd.Series) -> pd.Series:
    return values.where(np.isfinite(values) & (values > 0))
