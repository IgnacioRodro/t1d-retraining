"""Plain-text extraction of dataset documentation (ReadMe, glossaries, licences).

JAEB datasets ship their table descriptions and mandatory attribution text as
RTF files. Script 01 converts them to plain text so they can be read and shared
together with the inventory.
"""

from __future__ import annotations

import re
from pathlib import Path

from .sniff import detect_encoding

DOC_NAME = re.compile(r"(readme|read_me|glossary|licen[cs]e|citation|data.?dictionary)", re.IGNORECASE)
DOC_EXTENSIONS = {".rtf", ".txt", ".md", ".cff"}
MAX_DOC_BYTES = 5_000_000


def is_documentation(path: Path) -> bool:
    """True for small README / glossary / licence files."""
    return (
        path.suffix.lower() in DOC_EXTENSIONS
        and bool(DOC_NAME.search(path.name))
        and path.stat().st_size <= MAX_DOC_BYTES
    )


def rtf_to_text(rtf: str) -> str:
    """Strip RTF markup; uses the ``striprtf`` package when available."""
    try:
        from striprtf.striprtf import rtf_to_text as strip

        return strip(rtf, errors="ignore")
    except ImportError:
        text = re.sub(r"\{\\(fonttbl|colortbl|stylesheet|info)([^{}]|\{[^{}]*\})*\}", "", rtf)
        text = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: bytes.fromhex(m.group(1)).decode("latin-1"), text)
        text = re.sub(r"\\(par|line)d?\b", "\n", text)
        text = re.sub(r"\\tab\b", "\t", text)
        text = re.sub(r"\{\\\*[^{}]*\}", "", text)
        text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
        text = re.sub(r"[{}]", "", text)
        return re.sub(r"\n{3,}", "\n\n", text)


def extract_text(path: Path) -> str:
    """Plain text of a documentation file."""
    encoding = detect_encoding(path)
    raw = path.read_text(encoding=encoding, errors="ignore")
    return rtf_to_text(raw) if path.suffix.lower() == ".rtf" else raw
