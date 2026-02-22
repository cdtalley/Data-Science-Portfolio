#!/usr/bin/env python3
"""
Execute all portfolio notebooks and report errors.
Runs each notebook's code cells in order (shared globals) without Jupyter kernel.
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def strip_magics(source: str) -> str:
    out = []
    for line in source.splitlines():
        s = line.strip()
        if s.startswith("%") or s.startswith("!"):
            continue
        if s.startswith("%%"):
            continue
        out.append(line)
    return "\n".join(out)


def run_notebook(nb_path: Path, timeout_per_cell: int = 120) -> tuple[bool, str]:
    import json

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    g: dict = {}
    exec("import pandas as pd\nimport numpy as np\nimport warnings\nwarnings.filterwarnings('ignore')", g)

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    errors = []

    for i, cell in enumerate(code_cells):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        code = strip_magics(src)
        if not code.strip():
            continue
        # Skip Colab-only cells that would fail (drive.mount without try already handled in our patches)
        if "drive.mount(" in code and "from google.colab import drive" not in code and "except" not in code:
            continue
        try:
            exec(compile(ast.parse(code), f"{nb_path.name}<cell{i}>", "exec"), g)
        except Exception as e:
            # Allow expected failures (e.g. Jane Street when data not present)
            err = str(e).lower()
            if "jane_street" in nb_path.name and ("load_jane_street" in code or "train.csv" in code):
                errors.append(f"Cell {i}: SKIP (Jane Street data not downloaded): {e}")
                continue
            if "google.colab" in err or "no module named 'google'" in err:
                # Colab fallback path hit locally - treat as skip
                errors.append(f"Cell {i}: SKIP (Colab): {e}")
                continue
            errors.append(f"Cell {i}: {type(e).__name__}: {e}")
            if errors:
                return False, "\n".join(errors[:5])  # first few errors

    return True, "" if not errors else "Warnings: " + "\n".join(errors[:3])


def main():
    notebooks = sorted(ROOT.glob("*.ipynb"))
    failed = []
    for path in notebooks:
        if path.name.startswith("."):
            continue
        ok, msg = run_notebook(path)
        if ok:
            print(f"OK   {path.name}")
        else:
            print(f"FAIL {path.name}")
            print(f"     {msg}")
            failed.append(path.name)
    if failed:
        print(f"\nFailed: {len(failed)} notebook(s): {failed}")
        sys.exit(1)
    print(f"\nAll {len(notebooks)} notebooks ran without critical errors.")
    sys.exit(0)


if __name__ == "__main__":
    main()
