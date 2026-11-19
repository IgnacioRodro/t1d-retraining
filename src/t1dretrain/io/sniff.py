"""Robust reading of raw tabular files whose format is not known in advance.

Public diabetes datasets mix encodings (UTF-8, UTF-8 with BOM, UTF-16) and
delimiters (comma, semicolon, pipe, tab). These helpers detect both, count rows
without loading the file, and read the first and last rows. The raw-data
inventory (script 01) uses them, and so will the dataset loaders.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pandas as pd

TEXT_TABLE_EXTENSIONS = {".csv", ".txt", ".tsv", ".dat"}
EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}
CANDIDATE_DELIMITERS = [",", ";", "|", "\t"]
_SAMPLE_BYTES = 64 * 1024
_CHUNK_BYTES = 16 * 1024 * 1024


def detect_encoding(path: Path) -> str:
    """Guess the text encoding from the byte-order mark and the first bytes."""
    with open(path, "rb") as handle:
        head = handle.read(_SAMPLE_BYTES)
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if head.startswith(b"\xff\xfe"):
        return "utf-16"
    if head.startswith(b"\xfe\xff"):
        return "utf-16"
    if head and head.count(b"\x00") > len(head) // 4:
        # many NUL bytes without a BOM: UTF-16, byte order from the first NUL
        return "utf-16-le" if head[1:2] == b"\x00" else "utf-16-be"
    try:
        head.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError as error:
        # a multi-byte character cut at the sample boundary is still UTF-8
        if error.start >= len(head) - 4:
            return "utf-8"
        return "latin-1"


def read_text_sample(path: Path, encoding: str, n_bytes: int = _SAMPLE_BYTES) -> str:
    """First characters of a text file, decoded."""
    with open(path, "rb") as handle:
        raw = handle.read(n_bytes)
    if encoding.startswith("utf-16") and len(raw) % 2:
        raw = raw[:-1]
    return raw.decode(encoding, errors="ignore")


def detect_delimiter(sample: str) -> str | None:
    """Delimiter that splits the first lines into a constant number (> 1) of fields."""
    lines = [line for line in sample.splitlines()[:30] if line.strip()]
    if len(lines) > 1:
        lines = lines[:-1]  # the last sampled line may be cut
    if not lines:
        return None
    best, best_score = None, 0
    for delimiter in CANDIDATE_DELIMITERS:
        counts = [len(next(csv.reader([line], delimiter=delimiter))) for line in lines]
        mode, frequency = Counter(counts).most_common(1)[0]
        if mode < 2:
            continue
        score = frequency * mode
        if score > best_score:
            best, best_score = delimiter, score
    return best


def _is_number(text: str) -> bool:
    try:
        float(text.replace(",", "."))
    except ValueError:
        return False
    return True


def detect_header_offset(sample: str, delimiter: str, max_lines: int = 40) -> int:
    """Index of the header line, skipping report preambles.

    Device exports (FreeStyle Libre, CareLink) put a few report lines before the
    real header. The header is taken as the first line with the largest number
    of non-empty, mostly non-numeric fields.
    """
    lines = sample.splitlines()[:max_lines]
    if len(lines) > 1:
        lines = lines[:-1]  # the last sampled line may be cut
    best_index, best_count = 0, -1
    for index, line in enumerate(lines):
        fields = [f.strip() for f in next(csv.reader([line], delimiter=delimiter), [])]
        non_empty = [f for f in fields if f]
        if len(non_empty) < 2:
            continue
        if sum(_is_number(f) for f in non_empty) / len(non_empty) > 0.5:
            continue
        if len(non_empty) > best_count:
            best_index, best_count = index, len(non_empty)
    return best_index


def count_lines(path: Path, encoding: str) -> int:
    """Number of lines, read in chunks so that multi-GB files do not fill memory."""
    total = 0
    if encoding.startswith("utf-16"):
        with open(path, encoding=encoding, errors="ignore") as handle:
            while chunk := handle.read(_CHUNK_BYTES):
                total += chunk.count("\n")
        return total
    last = b""
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            total += chunk.count(b"\n")
            last = chunk[-1:]
    if last and last != b"\n":
        total += 1  # final line without newline
    return total


def read_head(path: Path, encoding: str, delimiter: str, n_rows: int = 5, header_offset: int = 0) -> pd.DataFrame:
    """First rows of a delimited text file, every column read as text."""
    return pd.read_csv(
        path,
        sep=delimiter,
        encoding=encoding,
        nrows=n_rows,
        skiprows=header_offset,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )


def read_tail(path: Path, encoding: str, delimiter: str, columns: list[str], n_rows: int = 5) -> pd.DataFrame:
    """Last rows of a delimited text file, read from the end of the file."""
    size = path.stat().st_size
    with open(path, "rb") as handle:
        handle.seek(max(0, size - _SAMPLE_BYTES))
        raw = handle.read()
    if encoding.startswith("utf-16"):
        if len(raw) % 2:
            raw = raw[1:]
        if encoding == "utf-16":  # BOM only at file start; assume little endian afterwards
            encoding = "utf-16-le"
    text = raw.decode(encoding, errors="ignore")
    lines = [line for line in text.splitlines() if line.strip()]
    if size > _SAMPLE_BYTES and lines:
        lines = lines[1:]  # first line of the window may be cut
    rows = list(csv.reader(lines[-n_rows:], delimiter=delimiter))
    rows = [row for row in rows if len(row) == len(columns)]
    return pd.DataFrame(rows, columns=columns)


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read a whole delimited text file with the detected encoding, delimiter and header line."""
    encoding = detect_encoding(path)
    sample = read_text_sample(path, encoding)
    delimiter = detect_delimiter(sample) or ","
    kwargs.setdefault("skiprows", detect_header_offset(sample, delimiter))
    return pd.read_csv(path, sep=delimiter, encoding=encoding, **kwargs)
