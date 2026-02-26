"""Execute all project notebooks in place so outputs are saved. Uses project venv and PYTHONPATH."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "Modern_Classification_Workflow_Bankruptcy.ipynb",
    "Supervised_Learning_Heart_Disease_Prediction_using_Patient_Biometric_Data.ipynb",
    "Supervised_Learning_Capstone-Predicting_Telecom_Customer_Churn_(IBM_Watson_Analytics).ipynb",
    "Unsupervised_Learning_Capstone_New_York_City_Bus_Data.ipynb",
    "NJ_Transit_+_Amtrak_(NEC)_Rail_Performance_Business_Solution.ipynb",
    "Corporate_Bankruptcy_Prediction.ipynb",
    "Jane_Street_Market_Prediction_XGBoost_and_Hyperparameter_Tuning.ipynb",
]

def main():
    python = ROOT / "venv" / "Scripts" / "python.exe"
    if not python.exists():
        print("venv not found; create it first.", file=sys.stderr)
        return 1
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT)}
    failed = []
    for name in NOTEBOOKS:
        path = ROOT / name
        if not path.exists():
            print(f"Skip (missing): {name}")
            continue
        print(f"Running: {name} ...")
        r = subprocess.run(
            [str(python), "-m", "nbconvert", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=2400", str(path)],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=2700
        )
        if r.returncode != 0:
            print(f"  FAILED: {name}")
            if r.stderr:
                print(r.stderr[-2000:] if len(r.stderr) > 2000 else r.stderr)
            failed.append(name)
        else:
            print(f"  OK: {name}")
    if failed:
        print(f"\nFailed: {failed}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
