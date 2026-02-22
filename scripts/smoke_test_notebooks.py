#!/usr/bin/env python3
"""
Smoke test: run first N code cells of each notebook to verify imports + data load.
Fast; does not run full ML pipelines.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MAX_CELLS = 6  # imports + data load + 1 EDA cell (keeps test fast; NYC Bus has 6.7M rows)


def strip_magics(source: str) -> str:
    out = []
    for line in source.splitlines():
        s = line.strip()
        if s.startswith("%") or s.startswith("!"):
            continue
        out.append(line)
    return "\n".join(out)


def run_smoke(nb_path: Path) -> tuple[bool, str]:
    import json

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    g = {}
    exec("import pandas as pd\nimport numpy as np\nimport warnings\nwarnings.filterwarnings('ignore')", g)

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"][:MAX_CELLS]
    for i, cell in enumerate(code_cells):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        code = strip_magics(src)
        if not code.strip():
            continue
        if "drive.mount(" in code and "from google.colab import drive" not in code and "except" not in code:
            continue
        try:
            exec(compile(ast.parse(code), f"{nb_path.name}<{i}>", "exec"), g)
        except Exception as e:
            err = str(e).lower()
            if "jane_street" in nb_path.name and ("load_jane_street" in code or "train.csv" in code):
                return True, "SKIP (no Jane Street data)"
            if "google.colab" in err or "no module named 'google'" in err:
                return True, "SKIP (Colab path)"
            if isinstance(e, ModuleNotFoundError) and "tensorflow" in err:
                continue  # skip cell (TensorFlow optional)
            return False, f"Cell {i}: {type(e).__name__}: {e}"
    return True, "OK"


def main():
    notebooks = sorted(ROOT.glob("*.ipynb"))
    failed = []
    for path in notebooks:
        if path.name.startswith("."):
            continue
        ok, msg = run_smoke(path)
        status = "OK" if ok else "FAIL"
        print(f"  {status}  {path.name}  {msg}")
        if not ok:
            failed.append(path.name)
    if failed:
        print(f"\nFailed: {failed}")
        sys.exit(1)
    print(f"\nSmoke test passed for all {len(notebooks)} notebooks.")
    sys.exit(0)


if __name__ == "__main__":
    main()
