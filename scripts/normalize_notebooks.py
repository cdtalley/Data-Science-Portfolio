"""Strip execution_count and outputs from markdown cells so nbconvert accepts notebooks."""
from pathlib import Path
import json
import sys

def main():
    root = Path(__file__).resolve().parent.parent
    for path in root.glob("*.ipynb"):
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        changed = False
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "markdown":
                for key in ("execution_count", "outputs"):
                    if key in cell:
                        del cell[key]
                        changed = True
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"Normalized: {path.name}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
