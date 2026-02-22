"""
Verify that the senior analysis sections in each notebook execute correctly.
Tests the core pipeline + CV + metrics flow for each project.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from portfolio_utils import set_seed, load_bankruptcy, load_telecom_churn, load_heart

set_seed(42)
ok = 0
fail = 0


def test_bankruptcy():
    global ok, fail
    try:
        df = load_bankruptcy()
        y = df["Bankrupt?"]
        X = df.drop(columns=["Bankrupt?"]).select_dtypes(include=[np.number])
        X = X.loc[:, X.nunique() > 1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k=30)),
            ("clf", xgb.XGBClassifier(random_state=42, eval_metric="logloss")),
        ])
        scoring = {"f1": "f1_weighted", "roc_auc": "roc_auc"}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"  Bankruptcy: F1={cv_res['test_f1'].mean():.4f}, AUC={auc:.4f} OK")
        ok += 1
    except Exception as e:
        print(f"  Bankruptcy: FAIL - {e}")
        fail += 1


def test_churn():
    global ok, fail
    try:
        df = load_telecom_churn()
        drop_cols = [c for c in ["customerID", "CustomerID", "TotalCharges",
                     "Churn Label", "Churn Value", "Churn Score", "CLTV",
                     "Churn Reason", "Count", "Country", "State", "City",
                     "Zip Code", "Lat Long", "Latitude", "Longitude"] if c in df.columns]
        y = df["Churn"].map({"Yes": 1, "No": 0}) if df["Churn"].dtype == "object" else df["Churn"]
        X_raw = df.drop(columns=["Churn"] + drop_cols, errors="ignore")
        X = pd.get_dummies(X_raw, drop_first=True).astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", xgb.XGBClassifier(random_state=42, eval_metric="logloss",
                                       scale_pos_weight=len(y[y==0])/len(y[y==1]))),
        ])
        scoring = {"f1": "f1", "roc_auc": "roc_auc"}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"  Churn: F1={cv_res['test_f1'].mean():.4f}, AUC={auc:.4f} OK")
        ok += 1
    except Exception as e:
        print(f"  Churn: FAIL - {e}")
        fail += 1


def test_heart():
    global ok, fail
    try:
        df = load_heart()
        target_col = "target" if "target" in df.columns else "output"
        y = df[target_col]
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ])
        scoring = {"f1": "f1", "roc_auc": "roc_auc"}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"  Heart: F1={cv_res['test_f1'].mean():.4f}, AUC={auc:.4f} OK")
        ok += 1
    except Exception as e:
        print(f"  Heart: FAIL - {e}")
        fail += 1


if __name__ == "__main__":
    print("Verifying senior analysis sections...")
    test_bankruptcy()
    test_churn()
    test_heart()
    print(f"\nResults: {ok} passed, {fail} failed")
