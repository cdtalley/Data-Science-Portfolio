"""
Install numpy and core notebook deps into the Python that runs this script.
Run with the SAME Python your Jupyter kernel uses, so the kernel sees the packages.

From repo root:

  # If you use the project venv (recommended):
  .\venv\Scripts\python.exe scripts/install_deps.py

  # Or activate venv first then:
  python scripts/install_deps.py
"""
import subprocess
import sys

def main():
    python = sys.executable
    core = [
        "numpy",
        "pandas",
        "openpyxl",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "scipy",
        "imbalanced-learn",
        "jupyter",
        "notebook",
        "kaggle",
    ]
    print(f"Using: {python}")
    print("Installing:", " ".join(core))
    subprocess.check_call([python, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    subprocess.check_call([python, "-m", "pip", "install", "-q"] + core)
    print("Done. In Jupyter: Kernel → Select Kernel → choose this interpreter:")
    print(f"  {python}")

if __name__ == "__main__":
    main()
