"""Write src/t1dretrain/code_manifest.json (hash of every released Python file).

Run before packing a release:  python tools/build_code_manifest.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t1dretrain import __version__  # noqa: E402
from t1dretrain.codecheck import MANIFEST, build_manifest  # noqa: E402

manifest = build_manifest(__version__)
MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print(f"{MANIFEST}: release {__version__}, {len(manifest['files'])} files")
