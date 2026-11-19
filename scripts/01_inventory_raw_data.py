"""01 - Inventory of the raw dataset folders.

Walks every dataset folder listed in config/paths.yaml and describes each file:
format, encoding, delimiter, header line, number of rows and columns. Files that
differ only in numbers (one file per patient or per day) are grouped into a
"pattern"; the largest file of each pattern is read in detail (columns, first
and last rows). Documentation files (ReadMe, glossaries, licences) are converted
to plain text. The dataset loaders (script 02) are written from this inventory,
so the code never relies on assumptions about file layouts.

Files are described in parallel on all available cores except the reserve
(rule R5; see config/project.yaml and t1dretrain.compute).

Outputs (results/01_inventory_raw_data/):
    files.csv            one row per file (pattern, kind, size, encoding, delimiter, header line, rows)
    patterns.csv         one row per file pattern (number of files, total rows, representative file)
    columns.csv          columns of each pattern's representative file, with first and last values
    zip_members.csv      contents of any zip file still packed
    samples/<dataset>/   first and last rows of each pattern's representative (omit with --no-samples)
    docs/<dataset>/      plain-text copies of ReadMe, glossary and licence files
    overview.png         size and number of table patterns per dataset
    largest_patterns.png the largest file patterns by total rows
    summary.md           per-dataset digest: main patterns, their columns and time ranges
    run_manifest.json    run metadata, including the compute resources used

Usage:
    python scripts/01_inventory_raw_data.py                    # all datasets in config/paths.yaml
    python scripts/01_inventory_raw_data.py --datasets replace_bg dclp3
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain.envguard import ensure_project_env  # noqa: E402  (standard library only)

ensure_project_env(__file__)  # re-runs this script inside the project environment if needed

from t1dretrain.codecheck import verify_code  # noqa: E402  (standard library only)

verify_code(strict="--skip-code-check" not in sys.argv)  # stops if some files were not updated
if "--skip-code-check" in sys.argv:
    sys.argv.remove("--skip-code-check")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from t1dretrain.compute import detect_resources, parallel_map
from t1dretrain.config import CONFIG_DIR, load_paths
from t1dretrain.io.docs import extract_text, is_documentation
from t1dretrain.io.sniff import (
    EXCEL_EXTENSIONS,
    TEXT_TABLE_EXTENSIONS,
    count_lines,
    detect_delimiter,
    detect_encoding,
    detect_header_offset,
    read_head,
    read_tail,
    read_text_sample,
)
from t1dretrain.outputs import ScriptOutput
from t1dretrain.plotting import set_style

SCRIPT_ID = "01_inventory_raw_data"
TABLE_KINDS = ["table", "excel_sheet", "parquet"]
TIME_LIKE = re.compile(
    r"(time|date|stamp|hora|fecha|tiempo|temporal|dttm|dtm|daysfrom|hour|minute|^dt|dt$|tm$|^ts$|^day)",
    re.IGNORECASE,
)
COLUMNS_SCHEMA = ["dataset", "pattern", "file", "sheet", "position", "column", "time_like", "first_value", "last_value"]


def file_pattern(rel_path: str) -> str:
    """Path with every run of digits replaced by '#', so per-patient or per-day files group together."""
    return re.sub(r"\d+", "#", rel_path)


def safe_name(text: str) -> str:
    """File-system friendly version of a relative path."""
    return re.sub(r"[^A-Za-z0-9._-]+", "__", text)


# ----------------------------------------------------------------------------- per-file description


def describe_text_table(path: Path, n_sample: int, detailed: bool):
    """Encoding, delimiter, header line and row count; head and tail when detailed."""
    encoding = detect_encoding(path)
    sample = read_text_sample(path, encoding)
    delimiter = detect_delimiter(sample)
    info = {"encoding": encoding, "delimiter": {"\t": "tab"}.get(delimiter, delimiter)}
    if delimiter is None:
        info.update(kind="text", n_rows=count_lines(path, encoding))
        return info, None, None
    offset = detect_header_offset(sample, delimiter)
    info.update(kind="table", header_line=offset + 1, n_rows=max(count_lines(path, encoding) - offset - 1, 0))
    if not detailed:
        return info, None, None
    head = read_head(path, encoding, delimiter, n_sample, header_offset=offset)
    tail = read_tail(path, encoding, delimiter, list(head.columns), n_sample)
    info["n_cols"] = head.shape[1]
    return info, head, tail


def describe_excel(path: Path, n_sample: int):
    """One description per worksheet of an Excel workbook."""
    results = []
    workbook = pd.ExcelFile(path)
    for index, sheet in enumerate(workbook.sheet_names):
        head = workbook.parse(sheet, nrows=n_sample, dtype=str)
        try:
            from openpyxl import load_workbook

            n_rows = load_workbook(path, read_only=True).worksheets[index].max_row - 1
        except Exception:  # noqa: BLE001 - the row count is optional
            n_rows = None
        results.append(({"kind": "excel_sheet", "sheet": sheet, "n_rows": n_rows, "n_cols": head.shape[1]}, head, None))
    return results


def describe_parquet(path: Path, n_sample: int):
    """Row count and first rows of a parquet file."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    head = parquet.read_row_group(0).to_pandas().head(n_sample).astype(str)
    return {"kind": "parquet", "n_rows": parquet.metadata.num_rows, "n_cols": head.shape[1]}, head, None


def describe_zip(path: Path):
    """Members of a zip file that has not been extracted."""
    with zipfile.ZipFile(path) as archive:
        members = pd.DataFrame(
            [{"member": m.filename, "size_mb": round(m.file_size / 1e6, 3)} for m in archive.infolist() if not m.is_dir()]
        )
    info = {"kind": "zip", "n_rows": len(members), "note": "zip not extracted; unzip it inside the dataset folder"}
    return info, members


def describe_file(path: Path, n_sample: int, detailed: bool):
    """List of (info, head, tail) tuples (Excel gives one per sheet) and zip members if any."""
    suffix = path.suffix.lower()
    if suffix in TEXT_TABLE_EXTENSIONS:
        return [describe_text_table(path, n_sample, detailed)], None
    if suffix in EXCEL_EXTENSIONS:
        return (describe_excel(path, n_sample) if detailed else [({"kind": "excel_sheet"}, None, None)]), None
    if suffix == ".parquet":
        return [describe_parquet(path, n_sample)], None
    if suffix == ".zip":
        info, members = describe_zip(path)
        return [(info, None, None)], members
    return [({"kind": "other"}, None, None)], None


def column_rows(dataset: str, pattern: str, rel_path: str, sheet: str, head: pd.DataFrame, tail) -> list[dict]:
    """One record per column with its first and last observed values."""
    rows = []
    for position, column in enumerate(head.columns):
        values = head[column].dropna()
        first = values.iloc[0] if len(values) else ""
        last = ""
        if tail is not None and len(tail) and column in tail:
            tail_values = tail[column].dropna()
            last = tail_values.iloc[-1] if len(tail_values) else ""
        rows.append({"dataset": dataset, "pattern": pattern, "file": rel_path, "sheet": sheet,
                     "position": position, "column": column, "time_like": bool(TIME_LIKE.search(str(column))),
                     "first_value": str(first)[:60], "last_value": str(last)[:60]})
    return rows


def describe_task(task: dict) -> dict:
    """Worker: describe one file. Runs in a separate process (must stay at module level)."""
    path, record = Path(task["path"]), task["record"]
    result = {"files": [], "columns": [], "samples": [], "zip_members": None, "doc_text": None}
    try:
        described, members = describe_file(path, task["n_sample"], record["representative"])
    except Exception as error:  # noqa: BLE001 - one bad file must not stop the inventory
        described, members = [({"kind": "unreadable", "error": f"{type(error).__name__}: {error}"[:300]}, None, None)], None
    if members is not None and len(members):
        result["zip_members"] = members.assign(dataset=record["dataset"], zip_file=record["file"])
    for info, head, tail in described:
        sheet = info.get("sheet", "")
        result["files"].append({**record, **info})
        if head is None:
            continue
        result["columns"] += column_rows(record["dataset"], record["pattern"], record["file"], sheet, head, tail)
        if task["keep_samples"]:
            sample = head.assign(_part="head")
            if tail is not None and len(tail):
                sample = pd.concat([sample, tail.assign(_part="tail")], ignore_index=True)
            result["samples"].append((record["file"] + (f"__{sheet}" if sheet else ""), sample))
    if is_documentation(path):
        try:
            result["doc_text"] = extract_text(path)
        except Exception as error:  # noqa: BLE001
            result["doc_text"] = f"[could not extract text: {error}]"
    return result


# ----------------------------------------------------------------------------- orchestration


def build_tasks(dataset: str, root: Path, n_sample: int, keep_samples: bool) -> list[dict]:
    """One task per file; the largest file of each pattern is described in detail."""
    paths = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    print(f"[{dataset}] {len(paths)} files in {root}")
    base = pd.DataFrame({"path": [str(p) for p in paths],
                         "file": [str(p.relative_to(root)) for p in paths],
                         "size_mb": [round(p.stat().st_size / 1e6, 3) for p in paths]})
    if base.empty:
        return []
    base["pattern"] = base["file"].map(file_pattern)
    representatives = set(base.loc[base.groupby("pattern")["size_mb"].idxmax(), "file"])
    tasks = []
    for row in base.itertuples():
        record = {"dataset": dataset, "pattern": row.pattern, "file": row.file,
                  "representative": row.file in representatives,
                  "extension": Path(row.file).suffix.lower(), "size_mb": row.size_mb}
        tasks.append({"path": row.path, "record": record, "n_sample": n_sample, "keep_samples": keep_samples})
    # largest files first, so the slowest tasks start early and the workers stay busy
    return sorted(tasks, key=lambda t: -t["record"]["size_mb"])


def summarise_patterns(files: pd.DataFrame) -> pd.DataFrame:
    """One row per pattern: number of files, total size and rows, representative file."""
    grouped = files.groupby(["dataset", "pattern"], dropna=False)
    patterns = grouped.agg(
        n_files=("file", "count"),
        total_size_mb=("size_mb", "sum"),
        total_rows=("n_rows", lambda s: pd.to_numeric(s, errors="coerce").sum()),
        kind=("kind", lambda s: s.mode().iloc[0] if len(s.mode()) else ""),
    ).reset_index()
    reps = files[files["representative"] == True].drop_duplicates(["dataset", "pattern"])  # noqa: E712
    patterns = patterns.merge(reps[["dataset", "pattern", "file", "n_cols"]].rename(columns={"file": "representative_file"}),
                              on=["dataset", "pattern"], how="left")
    return patterns.sort_values(["dataset", "total_rows"], ascending=[True, False])


def plot_overview(files: pd.DataFrame, patterns: pd.DataFrame) -> plt.Figure:
    """Size on disk and number of table patterns per dataset."""
    size = files.groupby("dataset")["size_mb"].sum() / 1e3
    n_patterns = patterns[patterns["kind"].isin(TABLE_KINDS)].groupby("dataset").size().reindex(size.index, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(size.index, size.values, color="#457b9d")
    axes[0].set_ylabel("Size on disk (GB)")
    axes[0].set_title("Size per dataset", loc="left")
    axes[1].bar(n_patterns.index, n_patterns.values, color="#2a9d8f")
    axes[1].set_ylabel("Number of table patterns")
    axes[1].set_title("Distinct table layouts per dataset", loc="left")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    return fig


def plot_largest_patterns(patterns: pd.DataFrame, top: int = 25) -> plt.Figure:
    """The largest table patterns across all datasets, by total rows."""
    tables = patterns[patterns["kind"].isin(TABLE_KINDS) & (patterns["total_rows"] > 0)]
    tables = tables.sort_values("total_rows", ascending=False).head(top).iloc[::-1]
    labels = [f"{r.dataset}: {Path(r.pattern).name} (x{r.n_files})" for r in tables.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(tables) + 1.2))
    ax.barh(labels, tables["total_rows"].astype(float), color="#e76f51")
    ax.set_xscale("log")
    ax.set_xlabel("Total rows across files of the pattern (log scale)")
    ax.set_title(f"Largest {len(tables)} table patterns", loc="left")
    return fig


def build_summary(files: pd.DataFrame, patterns: pd.DataFrame, columns: pd.DataFrame, docs: list[str]) -> str:
    """Per-dataset digest: main patterns with their columns and time ranges."""
    lines = ["# 01 - Raw data inventory", ""]
    if docs:
        lines += [f"Documentation extracted to docs/: {', '.join(docs)}.", ""]
    for dataset, group in files.groupby("dataset"):
        kinds = group["kind"].value_counts().to_dict()
        n_patterns = int((patterns["dataset"] == dataset).sum())
        lines += [f"## {dataset}", "", f"- {len(group)} files in {n_patterns} patterns, "
                  f"{group['size_mb'].sum() / 1e3:.2f} GB; kinds: {kinds}."]
        if group["error"].notna().any():
            lines.append(f"- Unreadable files: {int(group['error'].notna().sum())} (see files.csv).")
        if (group["kind"] == "zip").any():
            lines.append("- Contains zip files that are still packed; unzip them and run this script again.")
        if (pd.to_numeric(group["header_line"], errors="coerce") > 1).any():
            lines.append("- Some tables have report lines before the header (see header_line in files.csv).")
        table_patterns = patterns[(patterns["dataset"] == dataset) & patterns["kind"].isin(TABLE_KINDS)].head(20)
        if len(table_patterns):
            lines += ["", "| Pattern | Files | Total rows | Columns (first 12) | Time-like columns: first -> last |",
                      "|---|---|---|---|---|"]
        for row in table_patterns.itertuples():
            cols = columns[(columns["dataset"] == dataset) & (columns["pattern"] == row.pattern)]
            names = ", ".join(cols["column"].astype(str).head(12))
            times = cols[cols["time_like"]]
            time_text = "; ".join(f"{c.column}: {c.first_value} -> {c.last_value}" for c in times.head(3).itertuples())
            rows_text = f"{int(row.total_rows):,}" if pd.notna(row.total_rows) else "?"
            lines.append(f"| {row.pattern} | {row.n_files} | {rows_text} | {names} | {time_text} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths", type=Path, default=CONFIG_DIR / "paths.yaml", help="dataset paths YAML")
    parser.add_argument("--datasets", nargs="*", help="subset of dataset keys (default: all configured)")
    parser.add_argument("--sample-rows", type=int, default=5, help="rows kept from the start and end of each table")
    parser.add_argument("--no-samples", action="store_true", help="do not write the sample rows")
    args = parser.parse_args()

    set_style()
    resources = detect_resources()
    print(f"Compute plan: {resources.describe()}")
    out = ScriptOutput(SCRIPT_ID, resources=resources)
    paths = load_paths(args.paths)
    selected = args.datasets or list(paths)

    tasks, missing = [], []
    for dataset in selected:
        root = paths.get(dataset)
        if root is None or not root.exists():
            print(f"[{dataset}] folder not found: {root}")
            missing.append({"dataset": dataset, "pattern": "", "file": str(root), "representative": False,
                            "kind": "missing_folder", "size_mb": 0.0})
            continue
        tasks += build_tasks(dataset, root, args.sample_rows, not args.no_samples)

    results = parallel_map(describe_task, tasks, resources.n_workers, label="files")

    all_files, all_columns, all_zip, docs = list(missing), [], [], []
    for task, result in zip(tasks, results):
        dataset = task["record"]["dataset"]
        all_files += result["files"]
        all_columns += result["columns"]
        if result["zip_members"] is not None:
            all_zip.append(result["zip_members"])
        for name, sample in result["samples"]:
            out.csv(sample, f"samples/{dataset}/{safe_name(name)}.csv")
        if result["doc_text"] is not None:
            target = f"docs/{dataset}/{safe_name(task['record']['file'])}.txt"
            out.text(result["doc_text"], target)
            docs.append(f"{dataset}/{task['record']['file']}")

    files = pd.DataFrame(all_files)
    for name in ["n_rows", "n_cols", "header_line", "encoding", "delimiter", "error"]:
        if name not in files:
            files[name] = pd.NA
    files = files.sort_values(["dataset", "file"]).reset_index(drop=True)
    columns = pd.DataFrame(all_columns, columns=COLUMNS_SCHEMA)
    patterns = summarise_patterns(files)

    out.csv(files, "files.csv")
    out.csv(patterns, "patterns.csv")
    out.csv(columns, "columns.csv")
    out.csv(pd.concat(all_zip, ignore_index=True) if all_zip else pd.DataFrame(), "zip_members.csv")
    out.figure(plot_overview(files, patterns), "overview.png")
    if (patterns["kind"].isin(TABLE_KINDS) & (patterns["total_rows"] > 0)).any():
        out.figure(plot_largest_patterns(patterns), "largest_patterns.png")
    out.text(build_summary(files, patterns, columns, docs), "summary.md")
    out.manifest(settings={"datasets": selected, "sample_rows": args.sample_rows, "samples": not args.no_samples})
    print(f"Results written to {out.dir}")


if __name__ == "__main__":
    main()
