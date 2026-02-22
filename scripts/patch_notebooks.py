"""
Patch all portfolio notebooks: replace Colab/Drive data loading with Kaggle/data-dir pattern.
Run from repo root.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch_corporate_bankruptcy():
    path = ROOT / "Corporate_Bankruptcy_Prediction.ipynb"
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if "drive.mount" in src and "Mounting Google drive" in src:
            c["source"] = [
                "# Load data: run setup_data.py once, or set DATA_DIR / use Kaggle API\n",
                "try:\n",
                "    from portfolio_utils.data_loader import load_bankruptcy\n",
                "    df = load_bankruptcy()\n",
                "except Exception:\n",
                "    import os\n",
                "    from pathlib import Path\n",
                "    path = Path(os.environ.get(\"DATA_DIR\", \"data\")) / \"corporate_bankruptcy\"\n",
                "    csvs = list(path.rglob(\"*.csv\")) if path.exists() else []\n",
                "    if csvs:\n",
                "        df = pd.read_csv(csvs[0])\n",
                "    else:\n",
                "        from google.colab import drive\n",
                "        drive.mount(\"/content/drive\", force_remount=False)\n",
                "        df = pd.read_csv(\"/content/drive/My Drive/Data/Company Bankruptcy Prediction/data.csv\")\n",
                "print(f\"Loaded shape: {df.shape}\")\n",
            ]
            c["outputs"] = []
            c["execution_count"] = None
        elif "read_csv" in src and "Company Bankruptcy" in src:
            c["source"] = ["# df loaded above\n", "df.head(2)\n"]
            c["outputs"] = []
            c["execution_count"] = None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Patched Corporate_Bankruptcy_Prediction.ipynb")


def patch_telecom_churn():
    path = ROOT / "Supervised_Learning_Capstone-Predicting_Telecom_Customer_Churn_(IBM_Watson_Analytics).ipynb"
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if "read_excel" in src and "WA_Fn-UseC" in src:
            c["source"] = [
                "# Load data: run setup_data.py once, or set DATA_DIR\n",
                "try:\n",
                "    from portfolio_utils.data_loader import load_telecom_churn\n",
                "    df = load_telecom_churn()\n",
                "except Exception:\n",
                "    import os\n",
                "    from pathlib import Path\n",
                "    path = Path(os.environ.get(\"DATA_DIR\", \"data\")) / \"telecom_churn\"\n",
                "    csvs = list(path.rglob(\"*.csv\")) if path.exists() else []\n",
                "    if csvs:\n",
                "        df = pd.read_csv(csvs[0])\n",
                "    else:\n",
                "        from google.colab import files\n",
                "        import io\n",
                "        uploaded = files.upload()\n",
                "        key = [k for k in uploaded if k.endswith(('.xlsx','.xls','.csv'))][0]\n",
                "        if key.endswith('.csv'):\n",
                "            df = pd.read_csv(io.BytesIO(uploaded[key]))\n",
                "        else:\n",
                "            df = pd.read_excel(io.BytesIO(uploaded[key]))\n",
                "print(f\"Loaded shape: {df.shape}\")\n",
            ]
            c["outputs"] = []
            c["execution_count"] = None
            break
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Patched Supervised_Learning_Capstone-Predicting_Telecom_Customer_Churn_(IBM_Watson_Analytics).ipynb")


def patch_nj_transit():
    path = ROOT / "NJ_Transit_+_Amtrak_(NEC)_Rail_Performance_Business_Solution.ipynb"
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if "drive.mount" in src and "2020_05" not in src:
            c["source"] = [
                "# Load data: run setup_data.py once\n",
                "try:\n",
                "    from portfolio_utils.data_loader import load_nj_transit\n",
                "    df = load_nj_transit()\n",
                "except Exception:\n",
                "    from google.colab import drive\n",
                "    drive.mount(\"/content/drive\")\n",
                "    df = pd.read_csv(\"/content/drive/My Drive/Data/2020_05.csv\")\n",
                "print(f\"Loaded shape: {df.shape}\")\n",
            ]
            c["outputs"] = []
            c["execution_count"] = None
        elif "2020_05.csv" in src and "read_csv" in src:
            c["source"] = ["# df loaded above (re-run first load cell if needed)\n", "df.head(2)\n"]
            c["outputs"] = []
            c["execution_count"] = None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Patched NJ_Transit notebook")


def patch_heart():
    path = ROOT / "Supervised_Learning_Heart_Disease_Prediction_using_Patient_Biometric_Data.ipynb"
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if "drive.mount" in src:
            c["source"] = [
                "# Load data: run setup_data.py once\n",
                "try:\n",
                "    from portfolio_utils.data_loader import load_heart\n",
                "    df_heart = load_heart()\n",
                "except Exception:\n",
                "    from google.colab import drive\n",
                "    drive.mount(\"/content/drive\")\n",
                "    df_heart = pd.read_csv(\"/content/drive/My Drive/Data/Heart.csv\")\n",
                "print(f\"Loaded shape: {df_heart.shape}\")\n",
            ]
            c["outputs"] = []
            c["execution_count"] = None
        elif "Heart.csv" in src and "read_csv" in src:
            c["source"] = ["# df_heart loaded above\n", "df_heart.head(2)\n"]
            c["outputs"] = []
            c["execution_count"] = None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Patched Heart Disease notebook")


def patch_nyc_bus():
    path = ROOT / "Unsupervised_Learning_Capstone_New_York_City_Bus_Data.ipynb"
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if "mta_1706" in src and "read_csv" in src:
            c["source"] = [
                "%matplotlib inline\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import seaborn as sns\n",
                "from matplotlib import pyplot as plt\n",
                "from scipy import stats\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.decomposition import PCA\n",
                "from sklearn.cluster import KMeans\n",
                "from sklearn.manifold import TSNE\n",
                "import math\n",
                "import time\n",
                "from sklearn import metrics\n",
                "from itertools import product\n",
                "from math import floor\n",
                "from sklearn.cluster import DBSCAN\n",
                "import os\n",
                "\n",
                "try:\n",
                "    from portfolio_utils.data_loader import load_nyc_bus\n",
                "    df = load_nyc_bus()\n",
                "except Exception:\n",
                "    p = os.path.join(os.environ.get(\"DATA_DIR\", \"data\"), \"nyc_bus\", \"mta_1706.csv\")\n",
                "    df = pd.read_csv(p, on_bad_lines=\"skip\", low_memory=False)\n",
                "print(f\"Loaded shape: {df.shape}\")\n",
            ]
            c["outputs"] = []
            c["execution_count"] = None
            break
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Patched NYC Bus notebook")


def patch_jane_street():
    path = ROOT / "Jane_Street_Market_Prediction_XGBoost_and_Hyperparameter_Tuning.ipynb"
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if "drive.mount" in src:
            c["source"] = [
                "try:\n",
                "    from portfolio_utils.data_loader import load_jane_street\n",
                "    train, features = load_jane_street()\n",
                "except Exception:\n",
                "    from google.colab import drive\n",
                "    drive.mount(\"/content/drive\", force_remount=False)\n",
                "    train = pd.read_csv(\"/content/drive/My Drive/Data/train.csv\")\n",
                "    features = pd.read_csv(\"/content/drive/My Drive/Data/features.csv\")\n",
                "print(f\"Train shape: {train.shape}, Features shape: {features.shape}\")\n",
            ]
            c["outputs"] = []
            c["execution_count"] = None
        elif 'read_csv("/content/drive/My Drive/Data/train.csv")' in src or "train.csv" in src and "features.csv" not in src and "read_csv" in src:
            c["source"] = ["# train, features loaded above\n", "train.head(2)\n"]
            c["outputs"] = []
            c["execution_count"] = None
        elif "features.csv" in src and "read_csv" in src and "train" not in "".join(c.get("source", []))[:30]:
            c["source"] = ["# features loaded above\n", "features.head(2)\n"]
            c["outputs"] = []
            c["execution_count"] = None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Patched Jane Street notebook")


if __name__ == "__main__":
    patch_corporate_bankruptcy()
    patch_telecom_churn()
    patch_nj_transit()
    patch_heart()
    patch_nyc_bus()
    patch_jane_street()
    print("Done.")
