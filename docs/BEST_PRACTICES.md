# Best Practices Guide (2026)

This document explains **why** each practice is used in the reference notebook and across the portfolio. Implement these patterns in your own notebooks for reproducibility, correct validation, and interpretability.

---

## 1. Reproducibility: fix the random seed

**What:** Call `set_seed(42)` at the start of the notebook and pass `random_state=42` to any function that uses randomness (e.g. `train_test_split`, `RandomizedSearchCV`, and every estimator).

**Why:**  
Machine learning runs involve randomness (data shuffle, train/test split, bootstrap, stochastic algorithms). Without a fixed seed, you cannot reproduce the same metrics or model. Reviewers and recruiters should get the same numbers when they run your code. Fixing the seed is the first step to reproducible research and production debugging.

**How (in code):**
```python
from portfolio_utils import set_seed
set_seed(42)
# Then for every random component:
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = XGBClassifier(random_state=42)
```

---

## 2. Use a Pipeline for preprocessing + model

**What:** Put every step that learns from data (scaler, imputer, feature selector, model) inside a single `sklearn.pipeline.Pipeline`. Fit the pipeline once on training data; use it to transform and predict on test data.

**Why:**  
- **Avoids data leakage:** If you fit a scaler on the full dataset (or on test data by mistake), test metrics are optimistically biased. With a pipeline, `fit` is only ever called on training folds.  
- **Consistency:** The same transformation is applied to train and test; you never forget to scale test data.  
- **Cross-validation done right:** When you call `cross_validate(pipeline, X, y)`, each fold fits the whole pipeline on that fold’s train split and evaluates on the holdout. Preprocessing is never fitted on the evaluation fold.

**How (in code):**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectKBest(f_classif, k=30)),
    ("estimator", RandomForestClassifier(random_state=42)),
])
from sklearn.model_selection import cross_validate
scores = cross_validate(pipe, X, y, cv=5, scoring=["accuracy", "f1_weighted", "roc_auc_ovr"])
```

---

## 3. Stratified train/test split and cross-validation

**What:** For classification, use `stratify=y` in `train_test_split` and use `StratifiedKFold` (or `cv=5` with `cross_validate`, which defaults to stratified for classification) so that each split keeps the same class proportions as the full dataset.

**Why:**  
Imbalanced classes (e.g. few bankruptcies, many non-bankruptcies) can lead to folds with no positive class or very different proportions. Stratification keeps evaluation fair and stable. Accuracy alone can be misleading on imbalanced data; always report precision, recall, F1, and (where useful) ROC-AUC or PR-AUC.

**How (in code):**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

---

## 4. Report multiple metrics, not just accuracy

**What:** For classification, report at least: accuracy, precision (weighted or macro), recall, F1, and for binary/imbalanced problems ROC-AUC or average precision. Use `classification_report` and `confusion_matrix` on the test set.

**Why:**  
Accuracy can be high even when the minority class is never predicted (e.g. “predict no bankruptcy” always). Precision and recall for the positive class show how well you detect the event of interest. F1 balances the two. ROC-AUC summarizes ranking performance across thresholds. Together they give a complete picture and show you understand evaluation beyond a single number.

**How (in code):**
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
if has_binary_or_proba:
    print("ROC-AUC:", roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]))
```

---

## 5. Model interpretability with SHAP

**What:** Use SHAP (Shapley values) to explain which features drove each prediction (local) or matter most overall (global). For tree models use `shap.TreeExplainer`; for others use `KernelExplainer` on a sample.

**Why:**  
Stakeholders and regulators need to understand “why did the model say this?” SHAP provides a theoretically grounded way to attribute predictions to features. Summary plots (beeswarm, bar) show global importance; force plots show single-instance explanations. This aligns with 2026 expectations for explainable AI (XAI) in production and audits.

**How (in code):**  
After fitting a pipeline, extract the final estimator and the data it actually sees (transformed by previous steps). Then:
```python
import shap
# For a pipeline: get estimator and transformed training data
est = pipeline.named_steps["estimator"]
X_train_transformed = pipeline[:-1].transform(X_train)
explainer = shap.TreeExplainer(est, X_train_transformed)
shap_values = explainer.shap_values(X_train_transformed[:500])
shap.summary_plot(shap_values, X_train_transformed[:500], feature_names=feature_names, show=False)
```

---

## 6. Locked environments (pyproject.toml + lockfile)

**What:** Declare dependencies in `pyproject.toml` and use a lockfile (`uv.lock` with `uv`, or `pip-tools` with `requirements.txt` generated from `requirements.in`) so that `pip install` or `uv sync` installs exact versions.

**Why:**  
Different versions of pandas, sklearn, or xgboost can change results or break code. A lockfile ensures that anyone (or CI) gets the same environment and the same behavior. This is standard in 2026 for reproducible data science and MLOps.

**How:**  
Use `uv sync` in this repo (or `pip install -r requirements.txt`). Add new deps with `uv add package` or by editing `pyproject.toml` and re-running `uv lock`.

---

## 7. Where to apply these in this repo

- **Every notebook:** `set_seed(42)`, `random_state=42`, stratified splits, multiple metrics.  
- **At least one notebook per project:** A single pipeline (preprocess + model) and `cross_validate` on that pipeline.  
- **One or two notebooks:** A SHAP section (e.g. Bankruptcy, Churn) for interpretability.  
- **Repo level:** `pyproject.toml` and optional lockfile; README and IMPROVEMENTS.md reference this guide.

The notebook **`Modern_Classification_Workflow_Bankruptcy.ipynb`** implements all of the above on the Corporate Bankruptcy dataset as the reference.
