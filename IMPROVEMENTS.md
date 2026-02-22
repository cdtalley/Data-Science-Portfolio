# Portfolio Revival: 2026 Improvements & New Techniques

This doc outlines **confirmed data status**, **2026 best practices**, and a **concrete roadmap** to revive and modernize the repo.

---

## Data status

**All datasets load successfully** via `portfolio_utils.data_loader` (Kaggle API or local `data/`):

| Dataset           | Loader             | Shape / notes                          |
|------------------|--------------------|----------------------------------------|
| Corporate Bankruptcy | `load_bankruptcy()`   | (6819, 96)                             |
| Telecom Churn     | `load_telecom_churn()` | (7043, 34); columns normalized         |
| NJ Transit       | `load_nj_transit()`   | (98698, 13)                            |
| Heart Disease    | `load_heart()`        | (1025, 14)                             |
| NYC Bus          | `load_nyc_bus()`      | (6.7M, 17); use `on_bad_lines='skip'`  |
| Jane Street      | `load_jane_street()`  | Optional; requires competition accept  |

Run once: `python setup_data.py --no-jane-street`. Smoke test: `python scripts/smoke_test_notebooks.py`.

---

## 2026 techniques to adopt

### 1. Reproducibility & project hygiene

- **Pipelines**  
  Use `sklearn.pipeline.Pipeline` (or `make_pipeline`) so preprocessing (scaler, imputer, selector) is fitted only on train and applied consistently to test. Avoids data leakage and forgotten steps.
- **Seeds**  
  Set `np.random.seed`, `random.seed`, and estimator `random_state=` (and in TF: `tf.keras.utils.set_random_seed`) so runs are reproducible.
- **Locked envs**  
  Prefer `pyproject.toml` + lockfile (`uv.lock` with `uv`, or `pip-tools` with `requirements.in` → `requirements.txt`) so installs are reproducible.
- **Dataset versioning**  
  Optional: DVC or a simple `data/README.md` with dataset source URLs and download dates.

### 2. Model evaluation & validation

- **Stratified splits**  
  Use `StratifiedKFold` / `stratify=` in `train_test_split` for classification so train/test class balance is representative.
- **Multiple metrics**  
  Report precision, recall, F1, and (for imbalanced) PR-AUC or ROC-AUC; avoid relying only on accuracy.
- **Cross-validation**  
  Use `cross_validate` or `cross_val_predict` with a **pipeline** (preprocessing + model) so scaling/selection is not fitted on test folds.

### 3. Interpretability (XAI)

- **SHAP**  
  Add a short “Model interpretability” section in 1–2 notebooks (e.g. Bankruptcy, Churn) using `shap.TreeExplainer` for tree models or `shap.KernelExplainer` for others; `shap.summary_plot` and `shap.force_plot` for one-off explanations. Keeps the narrative “why does the model say this?”.
- **Feature importance**  
  For tree models, use built-in `feature_importances_` plus a bar plot; mention that SHAP gives instance-level and more consistent attributions.

### 4. Code & API modernization

- **Pandas 2.x**  
  Use `on_bad_lines='skip'` (or `'warn'`) instead of deprecated `error_bad_lines` / `warn_bad_lines`.
- **sklearn**  
  Prefer `Pipeline`, `ColumnTransformer` for mixed types, and `cross_validate`; keep `GridSearchCV`/`RandomizedSearchCV` on the pipeline.
- **Type hints**  
  Use in `portfolio_utils` and any new modules (e.g. `def load_*(...) -> pd.DataFrame`).

### 5. MLOps-lite (optional)

- **Experiment logging**  
  Log key runs (params + metrics) to a small JSON/CSV or to MLflow for a single “flagship” notebook.
- **CI**  
  GitHub Action that runs `scripts/smoke_test_notebooks.py` (and optionally `setup_data.py` with a small subset) so notebooks stay runnable.
- **Containers**  
  Optional `Dockerfile` + `docker-compose` for “run everything in one command” for recruiters.

### 6. New content ideas (2026)

- **Small “from scratch” project**  
  One notebook using only `pyproject.toml` + `portfolio_utils`: load data → pipeline (preprocess + model) → cross_validate → SHAP summary. Shows clean structure.
- **RAG/LLM**  
  Keep the Streamlit RAG demo; optionally add a second app (e.g. simple API with FastAPI) or a notebook that uses the same embedding + retrieval logic.
- **Time series**  
  NJ Transit / rail could be extended with a proper time-based split and a simple forecast baseline (e.g. seasonal naive or small LSTM/Transformer) to show awareness of temporal leakage and modern TS tools.

---

## Implementation roadmap

| Priority | Task | Effort |
|----------|------|--------|
| High | Use `sklearn.pipeline.Pipeline` in at least one notebook (e.g. Bankruptcy or Churn) | 1–2 hrs |
| High | Add `set_seed()` + document in README; use consistent `random_state` in notebooks | ~30 min |
| High | Add `pyproject.toml` (and optional `uv`) so `pip install -e .` or `uv sync` works | ~30 min |
| Medium | Add SHAP section to Corporate Bankruptcy or Telecom Churn notebook | 1–2 hrs |
| Medium | Add `portfolio_utils.ml_utils` (e.g. `make_seed`, `build_classification_pipeline`) for reuse | ~1 hr |
| Medium | GitHub Action: smoke test on push (and optionally data setup) | ~1 hr |
| Low | DVC or `data/README.md` for dataset versioning | ~30 min |
| Low | One “clean” end-to-end notebook (load → pipeline → CV → SHAP) as reference | 2–3 hrs |

---

## Reference implementation (done)

- **[Modern_Classification_Workflow_Bankruptcy.ipynb](Modern_Classification_Workflow_Bankruptcy.ipynb)** — End-to-end example: `set_seed`, load data, stratified split, single pipeline (scaler → SelectKBest → XGBoost), `cross_validate` with multiple metrics, holdout evaluation, and SHAP summary plot. Each section has a short “why” explanation in markdown.
- **[docs/BEST_PRACTICES.md](docs/BEST_PRACTICES.md)** — Thorough explanation of each practice (reproducibility, pipelines, stratified splits, multiple metrics, SHAP, locked envs) and how to apply them in this repo.

---

## Quick wins already in place

- Kaggle API–backed data loaders; all five main datasets load successfully.
- `DATA_DIR` and Colab fallbacks keep notebooks runnable locally and in the cloud.
- Pandas 2–friendly options (e.g. `on_bad_lines='skip'` for NYC Bus).
- `display.max_columns` / `display.max_rows` fixes for pandas OptionError.
- Smoke test script to verify notebooks run without errors.
- README with setup and data verification command.

Use this file as a living checklist: tick items as you implement them and add new 2026 techniques as you adopt them.
