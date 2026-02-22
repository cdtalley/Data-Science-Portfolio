"""
Modernize all portfolio notebooks:
  1. Replace deprecated sns.distplot() with histplot / kdeplot
  2. Inject set_seed(42) into the first code cell of every notebook
  3. Update markdown comments referencing distplot
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NOTEBOOKS = [
    "Corporate_Bankruptcy_Prediction.ipynb",
    "Supervised_Learning_Capstone-Predicting_Telecom_Customer_Churn_(IBM_Watson_Analytics).ipynb",
    "NJ_Transit_+_Amtrak_(NEC)_Rail_Performance_Business_Solution.ipynb",
    "Supervised_Learning_Heart_Disease_Prediction_using_Patient_Biometric_Data.ipynb",
    "Unsupervised_Learning_Capstone_New_York_City_Bus_Data.ipynb",
    "Jane_Street_Market_Prediction_XGBoost_and_Hyperparameter_Tuning.ipynb",
]


def fix_distplot_line(line: str) -> str:
    """Convert a single sns.distplot() call to histplot or kdeplot."""
    if "distplot" not in line:
        return line
    if line.strip().startswith("#") or line.strip().startswith("//"):
        return line.replace("distplot", "histplot")

    if "hist = False" in line or "hist=False" in line:
        new = re.sub(r"sns\.distplot\(", "sns.kdeplot(", line)
        new = re.sub(r",\s*hist\s*=\s*False", "", new)
        new = re.sub(r",\s*kde\s*=\s*True", "", new)
        return new

    new = re.sub(r"sns\.distplot\(", "sns.histplot(", line)
    new = re.sub(r",\s*kde\s*=\s*True", "", new)
    if "kde=True" not in new and "kde_kws" not in new:
        new = new.rstrip().rstrip(")")
        new += ", kde=True)\n"
    return new


def fix_distplot_in_notebook(nb):
    """Replace all distplot calls and references in a notebook."""
    changed = 0
    for cell in nb["cells"]:
        new_source = []
        for line in cell.get("source", []):
            fixed = fix_distplot_line(line) if "distplot" in line else line
            if "FutureWarning: `distplot`" in line:
                continue
            new_source.append(fixed)
            if fixed != line:
                changed += 1
        cell["source"] = new_source
        if changed:
            cell["outputs"] = []
            cell["execution_count"] = None
    return changed


def inject_set_seed(nb, filename):
    """Add set_seed(42) to the first code cell if not already present."""
    full_src = "".join(
        "".join(c.get("source", []))
        for c in nb["cells"]
        if c["cell_type"] == "code"
    )
    if "set_seed" in full_src:
        return False

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src_lines = cell.get("source", [])
        src = "".join(src_lines)
        if "import" in src or "matplotlib" in src:
            seed_lines = [
                "from portfolio_utils import set_seed\n",
                "set_seed(42)\n",
                "\n",
            ]
            cell["source"] = seed_lines + src_lines
            cell["outputs"] = []
            cell["execution_count"] = None
            return True
    return False


def process_notebook(filename):
    path = ROOT / filename
    if not path.exists():
        print(f"  SKIP (not found): {filename}")
        return
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    dp = fix_distplot_in_notebook(nb)
    ss = inject_set_seed(nb, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"  {filename}: distplot fixes={dp}, set_seed injected={ss}")


if __name__ == "__main__":
    print("Modernizing notebooks…")
    for nb_file in NOTEBOOKS:
        process_notebook(nb_file)
    print("Done.")
