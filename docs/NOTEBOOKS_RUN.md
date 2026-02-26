# Running All Notebooks (with outputs)

To (re)run every notebook so outputs are saved (for commit/push or viewing on GitHub/nbviewer):

## One-time setup

1. **Venv and deps** (from repo root):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements-core.txt
   pip install -r requirements.txt   # full stack: tensorflow, statsmodels, etc.
   ```

2. **Data**  
   Ensure `data/` is populated (e.g. `python setup_data.py --no-jane-street`).  
   Telecom Churn and Jane Street may need Kaggle API if data is missing.

## Run all notebooks

From repo root with `venv` activated (or use full path to `venv\Scripts\python.exe`):

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m nbconvert --execute --inplace --ExecutePreprocessor.timeout=1200 .\Modern_Classification_Workflow_Bankruptcy.ipynb
python -m nbconvert --execute --inplace --ExecutePreprocessor.timeout=1200 .\Supervised_Learning_Heart_Disease_Prediction_using_Patient_Biometric_Data.ipynb
# ... repeat for each .ipynb, or use the script below.
```

Or run the helper script (runs each notebook in sequence; 30 min timeout per notebook):

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\venv\Scripts\python.exe scripts/run_all_notebooks.py
```

## Fixes applied (Feb 2026)

- **Schema:** Markdown cells had `execution_count`/`outputs`; removed via `scripts/normalize_notebooks.py`.
- **sklearn:** `max_features='auto'` → `max_features='sqrt'` (invalid in recent sklearn) in Heart, NJ Transit, Corporate Bankruptcy, Telecom Churn.
- **Pandas:** `interpolate(method='pad')` → `.ffill()` in NYC Bus (pandas 3.x).
- **Modern_Classification_Workflow_Bankruptcy:** `roc_auc_score(...).round(4)` → `round(roc_auc_score(...), 4)`.
- **Heart notebook:** Colab-only display cell wrapped in try/except; added statsmodels, tensorflow, flask, graphviz to venv.
- **NJ Transit:** Added `import pandas as pd`, `seaborn as sns`, `matplotlib.pyplot as plt` in first code cell.

After running, commit the updated `.ipynb` files so outputs are in version control.
