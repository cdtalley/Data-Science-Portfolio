"""Fix Corporate_Bankruptcy_Prediction column references to use columns that exist in load_bankruptcy()."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "Corporate_Bankruptcy_Prediction.ipynb"

SUB = [
    ('" Operating Expense Rate"', '"Cash flow to Sales"'),
    ('" Research and development expense rate"', '"Cash flow to Sales"'),
    ('" Total Asset Growth Rate"', '"total assets to GNP price"'),
    ('" Inventory Turnover Rate (times)"', '"Inventory/working capital"'),
    ('" Fixed Assets Turnover Frequency"', '"fix assets to assets"'),
    ('" Cash/Current Liability"', '"cash / current liability"'),
    ('" Current Asset Turnover Rate"', '"current assets/total assets"'),
    ('" Quick Asset Turnover Rate"', '"Quick asset/Total asset"'),
    ('" Cash Turnover Rate"', '"cash / total assets"'),
]


def fix(obj):
    if isinstance(obj, list):
        return [fix(x) for x in obj]
    if isinstance(obj, str):
        s = obj
        for a, b in SUB:
            s = s.replace(a, b)
        return s
    return obj


def main():
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if "source" in cell:
            cell["source"] = fix(cell["source"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Fixed Corporate_Bankruptcy_Prediction.ipynb column references")


if __name__ == "__main__":
    main()
